#!/usr/bin/env python3
"""Read-only IR geometry diagnosis; run with the project's 3dgs Python.

Example:
  python -u tools/diagnose_geometry.py --model_path output/RUN \
      --iterations 20000 30000

Evaluates all train/test views for the four Gaussian/network combinations,
then compares both networks on identical sampled positions and camera inputs.
No optimizer, Probe fitting, training, or checkpoint conversion is performed.
"""

import argparse
import ast
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from arguments import ModelParams, PipelineParams, OptimizationParams
from gaussian_renderer import render
from scene import Scene, GaussianModel, GeoRefineModel
from train import (render_eval_view, quantize_like_save_image,
                   condition_geometry_inputs, estimate_temporal_jitter_interval)
from utils.image_utils import psnr


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def select_views(views, limit):
    if not limit or limit >= len(views):
        return list(views)
    return [views[i] for i in np.linspace(0, len(views) - 1, limit, dtype=int)]


def load_config(model_path, device):
    parser = argparse.ArgumentParser(add_help=False)
    ModelParams(parser, device)
    PipelineParams(parser)
    OptimizationParams(parser)
    defaults = vars(parser.parse_args([]))
    expression = ast.parse((model_path / "cfg_args").read_text(), mode="eval").body
    if not isinstance(expression, ast.Call) or not isinstance(expression.func, ast.Name) or expression.func.id != "Namespace" or expression.args:
        raise ValueError("Expected a saved Namespace in cfg_args")
    saved = {}
    for item in expression.keywords:
        if item.arg is None:
            raise ValueError("Expanded keywords are not supported in cfg_args")
        value = item.value
        if isinstance(value, ast.Call):
            name = ast.unparse(value.func)
            if name not in ("device", "torch.device"):
                raise ValueError("Unsupported cfg_args constructor: " + name)
            # Construct outside eval: PyTorch may lazily initialize its NumPy bridge.
            saved[item.arg] = torch.device(*[ast.literal_eval(v) for v in value.args],
                                           **{v.arg: ast.literal_eval(v.value) for v in value.keywords})
        else:
            saved[item.arg] = ast.literal_eval(value)
    cfg = argparse.Namespace(**{**defaults, **saved})
    cfg.model_path = str(model_path)
    cfg.load_model_path = ""
    cfg.source_path = str((ROOT / cfg.source_path).resolve())
    cfg.data_device = device
    cfg.load2gpu_on_the_fly = True
    if not cfg.eval or cfg.data_branch != "rgbt":
        raise ValueError("Requires an RGBT run with a held-out eval split")
    if cfg.is_6dof:
        raise ValueError("This diagnostic currently supports the additive geometry branch only")
    if getattr(cfg, "thermal_sh_degree", -1) not in (None, -1, cfg.sh_degree):
        raise ValueError("Use a checkpoint from the original shared SH implementation, not the IR2 experiment")
    unsupported = [k for k in ("ir_temporal_affine", "ir_temporal_pose",
                               "ir_eval_temporal_ensemble", "phys_eval_with_render")
                   if getattr(cfg, k, False)]
    if unsupported:
        raise ValueError("Unsupported eval corrections (would confound the swap): " + ", ".join(unsupported))
    return cfg


def load_network(model_path, iteration, cfg, device):
    multires = None if cfg.geometry_time_multires < 0 else cfg.geometry_time_multires
    model = GeoRefineModel(device, cfg.is_blender, cfg.is_6dof, time_multires=multires)
    path = model_path / "Geometry" / f"iteration_{iteration}" / "geometry.pth"
    model.geometry.load_state_dict(torch.load(path, map_location=device))
    model.geometry.eval()
    model.geometry.requires_grad_(False)
    return model


def field(model, xyz, view, cfg, fid=None):
    n = len(xyz)
    # Same inputs as training/evaluation; translation is camera_center.
    direction = torch.as_tensor(-view.R.T[:, 2].copy(), dtype=xyz.dtype, device=xyz.device)
    cam_pos, direction = condition_geometry_inputs(
        view.camera_center.expand(n, -1), direction.expand(n, -1), cfg.geometry_conditioning)
    t = view.fid if fid is None else fid
    return model.step(xyz, t.reshape(1, 1).expand(n, -1), cam_pos, direction)


def norm_stats(values):
    values = values.reshape(-1).float()
    return {"mean": float(values.mean()), "p95": float(torch.quantile(values, .95)),
            "max": float(values.max())}


def fixed_input_check(gaussians, networks, groups, cfg, extent, interval, args, out):
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    index = torch.randperm(len(gaussians.get_xyz), generator=generator)[:args.sample_points]
    index = index.to(gaussians.get_xyz.device)
    np.save(out / "sample_point_indices.npy", index.cpu().numpy())
    xyz = gaussians.get_xyz[index].detach()
    base_scale = gaussians.get_scaling[index].detach()
    rows = []
    print(f"[FIELD] fixed base=G{args.iterations[0]}, points={len(index)}, no time jitter", flush=True)
    for split, views in groups.items():
        for vi, view in enumerate(views):
            view.load2device(cfg.data_device)
            outputs = {}
            for iteration, model in networks.items():
                dxyz, drot, dscale = field(model, xyz, view, cfg)
                outputs[iteration] = (dxyz, drot, dscale)
                low = (view.fid - interval * .25).clamp(0, 1)
                high = (view.fid + interval * .25).clamp(0, 1)
                dlow = field(model, xyz, view, cfg, low)[0]
                dhigh = field(model, xyz, view, cfg, high)[0]
                row = {"split": split, "view": view.image_name, "network_iteration": iteration,
                       "fid": float(view.fid), "points": len(index),
                       "scale_delta_abs_mean": float(dscale.abs().mean()),
                       "rotation_delta_raw_l2_mean": float(drot.norm(dim=-1).mean()),
                       "effective_scale_nonpositive_fraction": float(((base_scale + dscale) <= 0).float().mean()),
                       "time_probe_low": float(low), "time_probe_high": float(high)}
                for prefix, values in [("displacement_over_extent", dxyz.norm(dim=-1) / extent),
                                       ("time_probe_displacement_over_extent", (dhigh - dlow).norm(dim=-1) / extent)]:
                    row.update({prefix + "_" + k: v for k, v in norm_stats(values).items()})
                rows.append(row)
            a, b = (outputs[i] for i in args.iterations)
            for row in rows[-2:]:
                row["network_swap_displacement_over_extent_mean"] = float((b[0] - a[0]).norm(dim=-1).mean() / extent)
                row["network_swap_scale_delta_abs_mean"] = float((b[2] - a[2]).abs().mean())
            view.load2device("cpu")
            if (vi + 1) % 10 == 0 or vi + 1 == len(views):
                print(f"[FIELD] {split} {vi + 1}/{len(views)}", flush=True)
    with open(out / "fixed_inputs.csv", "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return {f"N{iteration}_{split}": {
        key: float(np.mean([r[key] for r in rows if r["split"] == split and r["network_iteration"] == iteration]))
        for key in ("displacement_over_extent_mean", "displacement_over_extent_p95",
                    "time_probe_displacement_over_extent_mean", "scale_delta_abs_mean",
                    "rotation_delta_raw_l2_mean", "effective_scale_nonpositive_fraction")}
        for iteration in networks for split in groups if groups[split]}


def save_panels(images, rows, iterations, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    early, late = iterations
    combinations = [(early, early), (early, late), (late, early), (late, late)]
    test_rows = [r for r in rows if r["split"] == "test"]
    names = sorted({r["view"] for r in test_rows}, key=lambda name:
                   next(r["psnr"] for r in test_rows if r["view"] == name and r["gaussian_iteration"] == late and r["network_iteration"] == late)
                   - next(r["psnr"] for r in test_rows if r["view"] == name and r["gaussian_iteration"] == early and r["network_iteration"] == early))
    paths = []
    for rank, name in enumerate(names[:3]):
        gt = images[(name, "gt")].astype(np.float32) / 255
        fig, axes = plt.subplots(2, 5, figsize=(18, 6), constrained_layout=True)
        axes[0, 0].imshow(gt.mean(0), cmap="gray", vmin=0, vmax=1)
        axes[0, 0].set_title("GT")
        axes[1, 0].text(.05, .5, f"{name}\nAbsolute error\nShared range: 0 to 0.10\nAll views in per_view.csv", fontsize=11)
        for column, (g, n) in enumerate(combinations, 1):
            pred = images[(name, g, n)].astype(np.float32) / 255
            row = next(r for r in test_rows if r["view"] == name and r["gaussian_iteration"] == g and r["network_iteration"] == n)
            axes[0, column].imshow(pred.mean(0), cmap="gray", vmin=0, vmax=1)
            axes[0, column].set_title(f"G{g} / N{n}\nPSNR {row['psnr']:.4f}")
            axes[1, column].imshow(np.abs(pred - gt).mean(0), cmap="inferno", vmin=0, vmax=.1)
        for axis in axes.reshape(-1):
            axis.axis("off")
        path = out / f"worst_change_{rank + 1}_{name}.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        paths.append(path.name)
    return paths


def run(args):
    os.chdir(ROOT)
    model_path = Path(args.model_path).resolve()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA and the project's rasterizer are required")
    device = "cuda:0"
    torch.cuda.set_device(0)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cfg = load_config(model_path, device)
    if min(args.iterations) < cfg.warm_up:
        raise ValueError("Both checkpoints must be after geometry warm-up")
    out = Path(args.output).resolve() if args.output else ROOT / "output/material_validation" / (
        "geometry_diagnostic_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    source_path = Path(cfg.source_path)
    if out == model_path or model_path in out.parents or out == source_path or source_path in out.parents:
        raise ValueError("Diagnostic output must be outside the source run and dataset")
    required = [model_path / "cfg_args"] + [
        model_path / directory / f"iteration_{i}" / filename
        for i in args.iterations for directory, filename in
        [("point_cloud", "point_cloud.ply"), ("Geometry", "geometry.pth")]]
    # Saved best-model mirrors are never selected implicitly.
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    for name in ("TemporalAffine", "TemporalPose"):
        if (model_path / name).exists():
            raise ValueError(f"Unsupported additional checkpoint branch: {name}")
    protected = required + ([model_path / "result.txt"] if (model_path / "result.txt").exists() else [])
    protected += [ROOT / p for p in ["train.py", "gaussian_renderer/__init__.py",
                                   "scene/gaussian_model.py", "scene/geometry_model.py"]]
    before = {str(p): digest(p) for p in protected}
    out.mkdir(parents=True, exist_ok=False)
    metadata = {"status": "running", "model_path": str(model_path), "iterations": args.iterations,
                "seed": args.seed, "output": str(out), "protected_sha256_before": before,
                "notes": ["No optimizer, training, or Probe refit; RGB geometry is not swapped.",
                          "G includes all Gaussian parameters and SH, not only geometry.",
                          "Cross-checkpoint mismatch can reflect co-adaptation, not overfitting alone.",
                          "Fixed-input points come from the early checkpoint and are not visibility-filtered.",
                          "Time perturbation is a sensitivity proxy, not ground-truth physical motion."]}
    write_json(out / "summary.json", metadata)
    print(f"OUTPUT: {out}\nSOURCE: {model_path}\nDEVICE: {torch.cuda.get_device_name(0)}", flush=True)
    with torch.no_grad():
        early, late = args.iterations
        g0 = GaussianModel(cfg.sh_degree, device)
        scene = Scene(cfg, g0, load_iteration=early, shuffle=False)
        if not scene.has_rgbt or not g0.has_thermal_branch:
            raise ValueError("Checkpoint does not contain the expected IR branch")
        g1 = GaussianModel(cfg.sh_degree, device)
        g1.load_ply(str(model_path / "point_cloud" / f"iteration_{late}" / "point_cloud.ply"),
                    device, og_number_points=g0.og_number_points)
        if not g1.has_thermal_branch:
            raise ValueError("Late checkpoint does not contain a compatible IR branch")
        clouds = {early: g0, late: g1}
        networks = {i: load_network(model_path, i, cfg, device) for i in args.iterations}
        groups = {"test": select_views(scene.getTestCameras(), args.max_test_views),
                  "train": select_views(scene.getTrainCameras(), args.max_train_views)}
        if not all(groups.values()):
            raise ValueError("Both train and test camera groups must be nonempty")
        metadata["views"] = {k: [v.image_name for v in vs] for k, vs in groups.items()}
        metadata["all_views"] = {"test": len(groups["test"]) == len(scene.getTestCameras()),
                                 "train": len(groups["train"]) == len(scene.getTrainCameras())}
        interval = estimate_temporal_jitter_interval(scene.getTrainCameras(), "fid_median")
        background = torch.tensor([1., 1., 1.] if cfg.white_background else [0., 0., 0.], device=device)
        results, rows, images = {}, [], {}
        with open(out / "per_view.csv", "w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["split", "view", "gaussian_iteration",
                                                        "network_iteration", "psnr", "l1", "mse", "raw_mse"])
            writer.writeheader()
            for gi, cloud in clouds.items():
                for ni, network in networks.items():
                    for split, views in groups.items():
                        key = f"G{gi}_N{ni}_{split}"
                        print(f"[EVAL] {key}, views={len(views)}", flush=True)
                        batch = []
                        for vi, view in enumerate(views):
                            view.load2device(device)
                            prediction = render_eval_view(
                                view, cloud, render, (cfg, background), network, cfg, gi, device,
                                cfg.is_6dof, scene.cameras_extent, feature_set="thermal",
                                eval_temporal_interval=interval).clamp(0, 1)
                            gt = view.original_image.to(device).clamp(0, 1)
                            qpred, qgt = quantize_like_save_image(prediction), quantize_like_save_image(gt)
                            row = {"split": split, "view": view.image_name, "gaussian_iteration": gi,
                                   "network_iteration": ni,
                                   # image_utils.psnr treats dimension 0 as the batch.
                                   # Match training_report: one sample contains all three channels.
                                   "psnr": float(psnr(qpred.unsqueeze(0), qgt.unsqueeze(0)).mean()),
                                   "l1": float((qpred - qgt).abs().mean()),
                                   "mse": float((qpred - qgt).square().mean()),
                                   "raw_mse": float((prediction - gt).square().mean())}
                            if not all(np.isfinite(row[k]) for k in ("psnr", "l1", "mse", "raw_mse")):
                                raise FloatingPointError(f"Nonfinite evaluation result: {key}/{view.image_name}")
                            writer.writerow(row)
                            stream.flush()
                            rows.append(row)
                            batch.append(row)
                            if split == "test" and not args.no_panels:
                                images[(view.image_name, gi, ni)] = (qpred * 255).round().byte().cpu().numpy()
                                images[(view.image_name, "gt")] = (qgt * 255).round().byte().cpu().numpy()
                            view.load2device("cpu")
                            if (vi + 1) % 10 == 0 or vi + 1 == len(views):
                                print(f"  {vi + 1}/{len(views)} PSNR={np.mean([r['psnr'] for r in batch]):.5f}", flush=True)
                        results[key] = {k: float(np.mean([r[k] for r in batch])) for k in ("psnr", "l1", "mse", "raw_mse")}
                        results[key]["views"] = len(batch)
        metadata["metrics"] = results
        metadata["fixed_input_metrics"] = fixed_input_check(
            g0, networks, groups, cfg, max(float(scene.cameras_extent), 1e-8), interval, args, out)
        metadata["panels"] = [] if args.no_panels else save_panels(images, rows, args.iterations, out)
    text = (model_path / "result.txt").read_text() if (model_path / "result.txt").exists() else ""
    checks = []
    for step in args.iterations:
        matches = re.findall(r"\[ITER " + str(step) + r"\] Evaluating test Thermal: L1 (\S+) PSNR (\S+)", text)
        observed = results[f"G{step}_N{step}_test"]
        check = {"iteration": step, "available": bool(matches) and metadata["all_views"]["test"]}
        if check["available"]:
            expected_l1, expected_psnr = map(float, matches[-1])
            check.update(reference_psnr=expected_psnr, observed_psnr=observed["psnr"],
                         delta_psnr=observed["psnr"] - expected_psnr,
                         passed=abs(observed["psnr"] - expected_psnr) <= .003 and abs(observed["l1"] - expected_l1) <= 1e-5)
        checks.append(check)
    metadata["baseline_checks"] = checks
    metadata["protected_sha256_after"] = {str(p): digest(p) for p in protected}
    metadata["source_assets_unchanged"] = metadata["protected_sha256_after"] == before
    metadata["status"] = "complete" if metadata["source_assets_unchanged"] and all(
        not c["available"] or c["passed"] for c in checks) else "validation_failed"
    write_json(out / "summary.json", metadata)
    lines = ["# IR geometry checkpoint diagnosis", "", f"Source: `{model_path}`", "",
             f"Status: **{metadata['status']}**. Source assets unchanged: {metadata['source_assets_unchanged']}.", "",
             "G = Gaussian parameters including SH; N = IR geometry-network weights.",
             "Train metrics use the selected views (all views by default), not the five-view training-log subset.", "",
             "| G | N | Test PSNR | Train PSNR |", "|---|---|---:|---:|"]
    for gi in args.iterations:
        for ni in args.iterations:
            lines.append(f"| {gi} | {ni} | {results[f'G{gi}_N{ni}_test']['psnr']:.6f} | {results[f'G{gi}_N{ni}_train']['psnr']:.6f} |")
    lines += ["", "## Interpretation limits", "", *["- " + note for note in metadata["notes"]],
              "- A swap that improves held-out views implicates network evolution; a bad swap alone can reflect co-adaptation.",
              "- Large field/time sensitivity is a diagnostic signal, not proof of geometrical error.",
              "- Confirm causality with matched continuation runs, one updating geometry and one freezing it.", "",
              "## Baseline reproduction", "", "```json", json.dumps(checks, indent=2), "```", "",
              "Full results: per_view.csv, fixed_inputs.csv, summary.json.", ""]
    (out / "report.md").write_text("\n".join(lines))
    for line in lines[8:16]:
        print(line, flush=True)
    print(f"{metadata['status'].upper()}: {out}", flush=True)
    if metadata["status"] != "complete":
        raise RuntimeError("Baseline reproduction or source integrity check failed; inspect summary.json before interpreting swaps")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--iterations", type=int, nargs=2, default=[20000, 30000], metavar=("EARLY", "LATE"))
    parser.add_argument("--output", default="")
    parser.add_argument("--sample_points", type=int, default=4096)
    parser.add_argument("--max_train_views", type=int, default=0, help="0 = all train views")
    parser.add_argument("--max_test_views", type=int, default=0, help="0 = all test views")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_panels", action="store_true")
    args = parser.parse_args()
    if not 0 < args.iterations[0] < args.iterations[1] or args.sample_points < 1 or min(args.max_train_views, args.max_test_views) < 0:
        parser.error("Require 0 < EARLY < LATE, positive sample_points, and nonnegative view limits")
    run(args)


if __name__ == "__main__":
    main()
