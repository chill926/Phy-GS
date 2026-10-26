#!/usr/bin/env python3
"""Read-only base/final covariance and alpha-aware shape/error localization.

Matches the installed CUDA covariance formula (including unnormalized refined
quaternions), and validates it by rendering the same view with precomputed
covariances. Shape projections retain every other Gaussian as an occluder.
"""
import argparse
import copy
import csv
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from gaussian_renderer import render
from scene import Scene, GaussianModel
from train import compute_geo_fields, quantize_like_save_image
from utils.image_utils import psnr
from tools.diagnose_geometry import load_config, load_network, digest, write_json


def covariance(scales, quaternion):
    # R below is transpose of the column-major glm::mat3 in forward.cu.
    # Do NOT normalize: the installed rasterizer does not normalize either.
    w, x, y, z = quaternion.unbind(-1)
    rotation = torch.stack([
        1 - 2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
        2*(x*y+w*z), 1 - 2*(x*x+z*z), 2*(y*z-w*x),
        2*(x*z-w*y), 2*(y*z+w*x), 1 - 2*(x*x+y*y)], -1).reshape(-1, 3, 3)
    factor = rotation * scales[:, None, :]
    cov = factor @ factor.transpose(1, 2)
    packed = torch.stack([cov[:, 0, 0], cov[:, 0, 1], cov[:, 0, 2],
                          cov[:, 1, 1], cov[:, 1, 2], cov[:, 2, 2]], -1).contiguous()
    # SVD avoids the catastrophic cancellation of tiny covariance eigenvalues.
    axes = np.linalg.svd(factor.cpu().double().numpy(), compute_uv=False)[:, ::-1].copy()
    return axes, packed


def ratios(axes):
    a, b, c = axes.T
    return b / np.maximum(a, 1e-12), c / np.maximum(b, 1e-12), c / np.maximum(a, 1e-12)


def shape_stats(axes):
    thin, long, total = ratios(axes)
    row = {"N": len(axes)}
    for name, values in [("middle_over_short", thin), ("long_over_middle", long), ("long_over_short", total)]:
        for q in [.5, .95, .99]:
            row[f"{name}_p{int(q*100)}"] = float(np.quantile(values, q))
    for name, mask in [("elongated_gt10", long > 10), ("elongated_gt100", long > 100),
                       ("thin_gt100", thin > 100), ("needle_like", (long > 10) & (thin < 3)),
                       ("sheet_like", (thin > 10) & (long < 3))]:
        row[name + "_count"] = int(mask.sum())
        row[name + "_fraction"] = float(mask.mean())
    return row


class CovarianceView:
    def __init__(self, cloud, packed):
        self.cloud, self.packed = cloud, packed

    def __getattr__(self, name):
        return getattr(self.cloud, name)

    def get_covariance(self, scaling_modifier=1.):
        if scaling_modifier != 1.:
            raise ValueError("This check uses scale modifier 1")
        return self.packed


def association(contribution, delta):
    area = contribution > .05
    positive = np.maximum(delta, 0)
    def mean(values, mask):
        return float(values[mask].mean()) if mask.any() else None
    fraction = float(area.mean())
    positive_share = float(positive[area].sum() / max(float(positive.sum()), 1e-15))
    return {"coverage_above_005": fraction,
            "mean_mse_change_inside": mean(delta, area), "mean_mse_change_outside": mean(delta, ~area),
            "mean_positive_mse_change_inside": mean(positive, area),
            "mean_positive_mse_change_outside": mean(positive, ~area),
            "positive_error_increase_share": positive_share,
            "positive_increase_area_enrichment": positive_share / fraction if fraction > 0 else None,
            "alpha_weighted_mse_change": float((contribution*delta).sum() / max(float(contribution.sum()), 1e-15))}


def panel(view, pair, maps, delta, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    gt = pair[0]["gt"].mean(0)
    overlay = np.repeat(gt[..., None], 3, -1)
    red = maps["actual_elongated"]
    cyan = maps["actual_thin"]
    strength = np.maximum(red, cyan) * .7
    color = np.stack([red, cyan, cyan], -1) / np.maximum(np.maximum(red, cyan)[..., None], 1e-8)
    overlay = overlay * (1-strength[..., None]) + color * strength[..., None]
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), constrained_layout=True)
    content = [(gt, "GT", "gray", 0, 1),
               (pair[0]["pred"].mean(0), f"{pair[0]['iteration']} / PSNR {pair[0]['psnr']:.4f}", "gray", 0, 1),
               (pair[1]["pred"].mean(0), f"{pair[1]['iteration']} / PSNR {pair[1]['psnr']:.4f}", "gray", 0, 1),
               (delta, "MSE change (red = worse)", "coolwarm", -.003, .003),
               (red, "Final elongated contribution (c/b > 10)", "magma", 0, 1),
               (cyan, "Final thin contribution (b/a > 100)", "magma", 0, 1),
               (maps["new_elongated"], "New elongated contribution", "magma", 0, 1),
               (overlay, "GT overlay: elongated=red, thin=cyan", None, 0, 1)]
    for ax, (value, title, cmap, low, high) in zip(axes.reshape(-1), content):
        im = ax.imshow(value, cmap=cmap, vmin=low, vmax=high)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
        if cmap not in ("gray", None):
            fig.colorbar(im, ax=ax, shrink=.65)
    fig.suptitle(view.image_name)
    fig.savefig(out / f"shape_error_{view.image_name}.png", dpi=130)
    plt.close(fig)


def write_csv(path, rows):
    with open(path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_path", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--iterations", type=int, nargs=2, default=[20000, 30000])
    args = p.parse_args()
    os.chdir(ROOT)
    device = "cuda:0"
    source, out = Path(args.model_path).resolve(), Path(args.output).resolve()
    cfg = load_config(source, device)
    if cfg.compute_cov3D_python:
        raise ValueError("Expected the actual run to use the CUDA covariance path")
    if out == source or source in out.parents:
        raise ValueError("Output must be outside the source run")
    out.mkdir(parents=True, exist_ok=False)
    protected = [source / "cfg_args", source / "result.txt", ROOT / "train.py",
                 ROOT / "gaussian_renderer/__init__.py"] + [
        source / directory / f"iteration_{i}" / file for i in args.iterations
        for directory, file in [("point_cloud", "point_cloud.ply"), ("Geometry", "geometry.pth")]]
    before = {str(f): digest(f) for f in protected}
    metadata = {"status": "running", "model": str(source), "iterations": args.iterations,
                "protected_before": before,
                "notes": ["Shape thresholds are diagnostics, not physical validity criteria.",
                          "Actual axes include additive scale and unnormalized quaternion corrections.",
                          "Contribution maps preserve alpha compositing and all other occluders.",
                          "Pixel association is not causal evidence; edges/occlusion can confound it.",
                          "New-elongated point IDs assume no topology change between the two checkpoints.",
                          "Base freeze continuation still permits network scale corrections."]}
    write_json(out / "summary.json", metadata)
    print("OUTPUT:", out, flush=True)
    stats, associations, checks, metrics = [], [], [], []
    with torch.no_grad():
        g0 = GaussianModel(cfg.sh_degree, device)
        scene = Scene(cfg, g0, load_iteration=args.iterations[0], shuffle=False)
        g1 = GaussianModel(cfg.sh_degree, device)
        g1.load_ply(str(source / "point_cloud" / f"iteration_{args.iterations[1]}" / "point_cloud.ply"),
                    device, og_number_points=g0.og_number_points)
        if len(g0.get_xyz) != len(g1.get_xyz):
            raise ValueError("Point topology differs; cannot identify newly elongated points")
        clouds = [g0, g1]
        nets = [load_network(source, i, cfg, device) for i in args.iterations]
        bg = torch.tensor([1., 1., 1.] if cfg.white_background else [0., 0., 0.], device=device)
        black = torch.zeros_like(bg)
        base_axes = [np.sort(g.get_scaling.cpu().double().numpy(), axis=1) for g in clouds]
        for i, axes in zip(args.iterations, base_axes):
            stats.append({"iteration": i, "view": "all", "kind": "base_all", **shape_stats(axes)})
        for vi, view in enumerate(scene.getTestCameras()):
            view.load2device(device)
            pair = []
            for i, cloud, net in zip(args.iterations, clouds, nets):
                deform = compute_geo_fields(view, cloud, net, cfg, bg, cfg, i, lambda _: 0.,
                                            0., device, cfg.is_blender, "thermal", eval_mode=True)
                actual_scale = cloud.get_scaling + deform[2]
                actual_quat = cloud.get_rotation + deform[1]
                axes, packed = covariance(actual_scale, actual_quat)
                rendered = render(view, cloud, cfg, bg, *deform, device, cfg.is_6dof, feature_set="thermal")
                pred, gt = quantize_like_save_image(rendered["render"]), quantize_like_save_image(view.original_image)
                visible = rendered["visibility_filter"].cpu().numpy()
                if not visible.any():
                    raise RuntimeError("No visible Gaussian")
                base = base_axes[args.iterations.index(i)]
                stats.append({"iteration": i, "view": view.image_name, "kind": "base_same_visible", **shape_stats(base[visible])})
                stats.append({"iteration": i, "view": view.image_name, "kind": "actual_visible", **shape_stats(axes[visible])})
                # Verify the covariance reconstruction against the real renderer.
                precomputed = copy.copy(cfg)
                precomputed.compute_cov3D_python = True
                check_image = render(view, CovarianceView(cloud, packed), precomputed, bg, *deform,
                                     device, cfg.is_6dof, feature_set="thermal")["render"]
                diff = (check_image - rendered["render"]).abs()
                check = {"iteration": i, "view": view.image_name, "mean_abs_render_difference": float(diff.mean()),
                         "max_abs_render_difference": float(diff.max()),
                         "quaternion_norm_mean": float(actual_quat.norm(dim=-1)[rendered["visibility_filter"]].mean()),
                         "nonpositive_scale_fraction": float((actual_scale[rendered["visibility_filter"]] <= 0).float().mean())}
                checks.append(check)
                if check["mean_abs_render_difference"] > 1e-5 or check["max_abs_render_difference"] > .003:
                    write_json(out / "covariance_checks.json", checks)
                    raise RuntimeError("Reconstructed covariance does not reproduce the CUDA renderer")
                value = float(psnr(pred.unsqueeze(0), gt.unsqueeze(0)).mean())
                metrics.append({"iteration": i, "view": view.image_name, "psnr": value})
                pair.append({"iteration": i, "axes": axes, "deform": deform, "pred": pred.cpu().numpy(),
                             "gt": gt.cpu().numpy(), "psnr": value})
            thin0, long0, _ = ratios(pair[0]["axes"])
            thin1, long1, _ = ratios(pair[1]["axes"])
            _, base_long, _ = ratios(base_axes[1])
            masks = {"base_elongated": base_long > 10, "actual_elongated": long1 > 10,
                     "actual_thin": thin1 > 100, "new_elongated": (long1 > 10) & (long0 <= 10)}
            delta = ((pair[1]["pred"]-pair[1]["gt"])**2).mean(0) - ((pair[0]["pred"]-pair[0]["gt"])**2).mean(0)
            maps = {}
            for name, mask in masks.items():
                colors = torch.as_tensor(mask, device=device, dtype=torch.float32)[:, None].expand(-1, 3).contiguous()
                contribution = render(view, g1, cfg, black, *pair[1]["deform"], device, cfg.is_6dof,
                                      override_color=colors, feature_set="thermal")["render"][0].clamp(0, 1).cpu().numpy()
                maps[name] = contribution
                associations.append({"view": view.image_name, "mask": name, "selected_gaussians": int(mask.sum()),
                                     **association(contribution, delta)})
            panel(view, pair, maps, delta, out)
            np.savez_compressed(out / f"maps_{view.image_name}.npz", mse_change=delta,
                                **maps, actual_axes_early=pair[0]["axes"], actual_axes_late=pair[1]["axes"])
            print(f"[SHAPE] {vi+1}/{len(scene.getTestCameras())} {view.image_name}: "
                  f"PSNR {pair[0]['psnr']:.5f} -> {pair[1]['psnr']:.5f}; covariance validated", flush=True)
            view.load2device("cpu")
    write_csv(out / "shape_statistics.csv", stats)
    write_csv(out / "pixel_association.csv", associations)
    write_csv(out / "per_view_psnr.csv", metrics)
    metadata["covariance_checks"] = checks
    metadata["base_shape_statistics"] = [row for row in stats if row["kind"] == "base_all"]
    metadata["base_same_visible_mean_over_views"] = {str(i): {
        key: float(np.mean([row[key] for row in stats if row["iteration"] == i and row["kind"] == "base_same_visible"]))
        for key in stats[0] if key not in ("iteration", "view", "kind")}
        for i in args.iterations}
    metadata["actual_shape_mean_over_views"] = {str(i): {
        key: float(np.mean([row[key] for row in stats if row["iteration"] == i and row["kind"] == "actual_visible"]))
        for key in stats[0] if key not in ("iteration", "view", "kind")}
        for i in args.iterations}
    metadata["association_mean_over_views"] = {mask: {
        key: float(np.mean([row[key] for row in associations if row["mask"] == mask and row[key] is not None]))
        if any(row[key] is not None for row in associations if row["mask"] == mask) else None
        for key in associations[0] if key not in ("mask", "view")}
        for mask in masks}
    metadata["baseline_psnr"] = {str(i): float(np.mean([row["psnr"] for row in metrics if row["iteration"] == i])) for i in args.iterations}
    metadata["protected_after"] = {str(f): digest(f) for f in protected}
    metadata["source_assets_unchanged"] = metadata["protected_after"] == before
    metadata["status"] = "complete" if metadata["source_assets_unchanged"] else "validation_failed"
    write_json(out / "summary.json", metadata)
    if metadata["status"] != "complete":
        raise RuntimeError("Source integrity check failed")
    print("COMPLETE:", out, flush=True)


if __name__ == "__main__":
    main()
