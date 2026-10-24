#!/usr/bin/env python3
"""Export the existing PhysIR Probe without fitting or modifying a checkpoint.

Example (from the repository root):
  python thesis_scripts/render_probe_maps.py -m output/experiment/all --iteration 30000
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
import hashlib
import json
import shlex
import subprocess
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from gaussian_renderer import render
from scene import Scene, GaussianModel, GeoRefineModel
from train import compute_geo_fields
from utils.probe_maps import build_probe_fields, decode_field, encode_field
from utils.radiative_utils import extract_thermal_gaussian_target, PlanckLUT


LABELS = {
    "temperature_K": ("Probe temperature (K; uncalibrated)", "inferno"),
    "emissivity": ("Material emissivity at fitted T", "viridis"),
    "emissivity_ref": ("Reference emissivity at T_ref", "viridis"),
    "emissivity_slope": ("Emissivity slope (K^-1)", "coolwarm"),
    "emission_proxy": ("Blackbody emission proxy u", "magma"),
    "ambient_proxy": ("Global ambient proxy a", "viridis"),
    "self_emission_proxy": ("Self-emission proxy: epsilon * u", "magma"),
    "reflected_ambient_proxy": ("Reflected proxy: (1-epsilon) * a", "viridis"),
    "residual_proxy": ("Model residual delta (signed proxy)", "coolwarm"),
    "q_phys_proxy": ("Physical fit q_phys (proxy)", "gray"),
    "q_target_proxy": ("Thermal SH DC target (gray proxy)", "gray"),
    "fit_abs_error_proxy": ("Mean pointwise |q_phys-q_target|", "magma"),
    "accumulation": ("Opacity accumulation (visibility only)", "gray"),
}


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stats(value):
    value = np.asarray(value)
    value = value[np.isfinite(value)]
    if not value.size:
        return {"count": 0}
    return {"count": int(value.size), "min": float(value.min()),
            "max": float(value.max()), "mean": float(value.mean()),
            "p05": float(np.percentile(value, 5)), "p95": float(np.percentile(value, 95))}


def choose_iteration(model_path, requested):
    if requested >= 0:
        candidates = [requested]
    else:
        candidates = sorted((int(p.name.split("_")[-1]) for p in
                             (model_path / "point_cloud").glob("iteration_*")
                             if p.name.split("_")[-1].isdigit()), reverse=True)
    for iteration in candidates:
        folder = model_path / "point_cloud" / ("iteration_{}".format(iteration))
        if (folder / "point_cloud.ply").is_file() and (folder / "phys_probe.pt").is_file():
            return iteration, folder
    raise ValueError("No paired point_cloud.ply + phys_probe.pt at the requested checkpoint. "
                     "Do not reuse a Probe from a different iteration; select a paired checkpoint.")


def display_scales(fields, opt):
    scales = {}
    for name, value in fields.items():
        low, high = float(value.min()), float(value.max())
        if name == "temperature_K":
            low, high = float(opt.phys_probe_temp_min), float(opt.phys_probe_temp_max)
        elif name == "residual_proxy":
            high = max(abs(low), abs(high), 1e-6)
            low = -high
        elif name in ("emission_proxy", "q_phys_proxy", "q_target_proxy"):
            low, high = 0.0, 1.0
        elif name == "fit_abs_error_proxy":
            low, high = 0.0, max(high, 1e-6)
        if high - low < 1e-7:
            padding = max(abs(low) * 0.1, 1e-6) if name == "emissivity_slope" else max(abs(low) * 0.01, 0.001)
            low, high = low - padding, high + padding
        scales[name] = [low, high]
    scales["accumulation"] = [0.0, 1.0]
    return scales


def scalar_png(value, path, name, scales):
    cmap = plt.get_cmap(LABELS[name][1]).copy()
    cmap.set_bad("#e5e7eb")
    low, high = scales[name]
    plt.imsave(str(path), np.ma.masked_invalid(value), cmap=cmap, vmin=low, vmax=high)


def rgb_array(value):
    return value.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()


def overview(path, view_name, arrays, rgb_gt, tir_gt, tir_render, scales, iteration, probe_mode):
    fig, axes = plt.subplots(3, 4, figsize=(18, 11), constrained_layout=True)
    for ax, image, title in zip(axes.flat, (rgb_gt, tir_gt, tir_render),
                                ("Input RGB", "Input thermal palette", "Baseline thermal render")):
        ax.imshow(image)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    names = ("temperature_K", "emissivity", "self_emission_proxy",
             "reflected_ambient_proxy", "residual_proxy", "q_phys_proxy",
             "q_target_proxy", "fit_abs_error_proxy", "accumulation")
    for ax, name in zip(list(axes.flat)[3:], names):
        low, high = scales[name]
        cmap = plt.get_cmap(LABELS[name][1]).copy()
        cmap.set_bad("#e5e7eb")
        im = ax.imshow(np.ma.masked_invalid(arrays[name]), cmap=cmap, vmin=low, vmax=high)
        ax.set_title(LABELS[name][0], fontsize=10)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.035, pad=0.015)
    fig.suptitle("PhysIR-Splat | {} | iteration {} | {}\n"
                 "Temperature uses the normalized Probe LUT, not radiometric calibration."
                 .format(view_name, iteration, probe_mode), fontsize=14)
    fig.savefig(str(path), dpi=140)
    plt.close(fig)


def plot_gaussians(path, xyz, fields, scales):
    idx = np.linspace(0, len(xyz)-1, min(len(xyz), 18000), dtype=int)
    points = xyz[idx]
    center = np.median(points, axis=0)
    # Centering changes display coordinates only; NPZ retains original coordinates.
    points = points - center
    fig = plt.figure(figsize=(13, 6), constrained_layout=True)
    for panel, name in enumerate(("temperature_K", "emissivity"), 1):
        ax = fig.add_subplot(1, 2, panel, projection="3d")
        low, high = scales[name]
        im = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=fields[name][idx],
                        cmap=LABELS[name][1], vmin=low, vmax=high, s=1, alpha=0.7)
        ranges = np.ptp(points, axis=0)
        ax.set_box_aspect(np.maximum(ranges, 1e-3))
        ax.set_title(LABELS[name][0], fontsize=11)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        fig.colorbar(im, ax=ax, shrink=0.65, pad=0.06)
    fig.suptitle("Canonical Gaussian centers (COLMAP units) | sampled display, not a volumetric temperature measurement")
    fig.savefig(str(path), dpi=160)
    plt.close(fig)


def run():
    device = torch.device("cuda:0")
    parser = argparse.ArgumentParser(description=__doc__)
    model_group = ModelParams(parser, device, sentinel=True)
    pipe_group = PipelineParams(parser)
    parser.set_defaults(**{name: None for name in vars(pipe_group)})
    opt_group = OptimizationParams(parser, sentinel=True)
    parser.add_argument("--iteration", type=int, default=-1,
                        help="Default: latest checkpoint with its own Probe cache")
    parser.add_argument("--split", choices=("test", "train", "all"), default="test")
    parser.add_argument("--max-views", type=int, default=0, help="0 exports every view in each selected split")
    parser.add_argument("--min-accumulation", type=float, default=0.5)
    parser.add_argument("--output-dir", default=None, help="Must be a new directory")
    args = get_combined_args(parser, device)
    if not 0 < args.min_accumulation <= 1 or args.max_views < 0:
        parser.error("Require 0 < min-accumulation <= 1 and max-views >= 0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the repository rasterizer")
    torch.cuda.set_device(device)
    torch.manual_seed(0)
    np.random.seed(0)
    model_path = Path(args.model_path).resolve()
    iteration, checkpoint = choose_iteration(model_path, args.iteration)
    opt_defaults = vars(OptimizationParams(argparse.ArgumentParser())).copy()
    opt_defaults.update(vars(args))
    args = argparse.Namespace(**opt_defaults)
    opt = opt_group.extract(args)
    unsupported = [name for name in ("ir_temporal_pose", "ir_temporal_affine", "ir_eval_temporal_ensemble")
                   if getattr(opt, name, False)]
    if unsupported:
        raise ValueError("This baseline exporter does not implement these enabled corrections: {}".format(unsupported))
    dataset = model_group.extract(args)
    dataset.model_path = str(model_path)
    dataset.load_model_path = ""
    dataset.data_device = device
    dataset.load2gpu_on_the_fly = True
    pipeline = pipe_group.extract(args)
    for name, value in vars(PipelineParams(argparse.ArgumentParser())).items():
        if getattr(pipeline, name, None) is None:
            setattr(pipeline, name, value)
    requested_output = getattr(args, "output_dir", None)
    output = Path(requested_output).resolve() if requested_output else model_path / "physical_fields" / ("iteration_{}_{}".format(iteration, args.split))
    if output.exists():
        raise FileExistsError("Output exists; choose a new --output-dir: {}".format(output))
    source_files = [checkpoint / "point_cloud.ply", checkpoint / "phys_probe.pt", model_path / "cfg_args",
                    model_path / "Geometry" / ("iteration_{}".format(iteration)) / "geometry.pth"]
    for path in source_files:
        if not path.is_file():
            raise FileNotFoundError("Missing matching checkpoint asset: {}".format(path))
    hashes_before = {str(path): digest(path) for path in source_files}
    state = torch.load(str(checkpoint / "phys_probe.pt"), map_location=device)
    probe_metadata = state.get("probe_metadata", {})
    probe_mode = probe_metadata.get("mode", "original")
    if probe_metadata.get("temperature_model", "normalized_planck_lut") != "normalized_planck_lut":
        raise ValueError("Exporter needs a validator for this temperature model")
    for option, key in (("phys_probe_temp_min", "temperature_min_K"),
                        ("phys_probe_temp_max", "temperature_max_K"),
                        ("phys_probe_lambda_min_um", "wavelength_min_um"),
                        ("phys_probe_lambda_max_um", "wavelength_max_um")):
        if key in probe_metadata:
            setattr(opt, option, probe_metadata[key])
    if int(state.get("iteration", -1)) != iteration:
        raise ValueError("Probe iteration does not match the point-cloud iteration")

    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, device)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        if getattr(gaussians, "has_temporal_thermal_branch", False):
            raise ValueError("Temporal thermal branches need a dedicated physical model")
        n = int(gaussians.get_xyz.shape[0])
        fields = build_probe_fields(state, n)
        current_q = extract_thermal_gaussian_target(gaussians)["q_target"].reshape(-1)
        q_alignment_error = float((current_q - state["q_target"].reshape(-1)).abs().max())
        if q_alignment_error > 2e-5:
            raise ValueError("Probe target differs from saved thermal SH DC (max error {}); possible stale/reordered cache".format(q_alignment_error))
        lut = PlanckLUT(opt.phys_probe_temp_min, opt.phys_probe_temp_max,
                        opt.phys_probe_lambda_min_um, opt.phys_probe_lambda_max_um, device)
        lut_error = float((lut.norm_radiance_to_temp(fields["emission_proxy"]) - fields["temperature_K"]).abs().max())
        if lut_error > 0.002:
            raise ValueError("Saved temperature does not match the configured LUT: {} K".format(lut_error))
        geo = GeoRefineModel(device, dataset.is_blender, dataset.is_6dof,
                             time_multires=opt.geometry_time_multires if opt.geometry_time_multires >= 0 else None)
        geo.load_weights(str(model_path), iteration=iteration)
        geo.geometry.eval()
        output.mkdir(parents=True)
        field_arrays = {name: value.cpu().numpy() for name, value in fields.items()}
        xyz = gaussians.get_xyz.cpu().numpy()
        scales = display_scales(fields, opt)
        point_extras = {}
        if "material_ids" in state:
            point_extras["material_ids"] = state["material_ids"].cpu().numpy().reshape(-1)
        np.savez_compressed(str(output / "gaussian_fields.npz"), xyz=xyz,
                            opacity=gaussians.get_opacity.cpu().numpy().reshape(-1),
                            **dict(field_arrays, **point_extras))
        coefficients = {name: state[name].cpu().numpy() for name in
                        ("cluster_eps0", "cluster_k", "cluster_counts") if name in state}
        if coefficients:
            np.savez_compressed(str(output / "material_parameters.npz"), **coefficients)
        plot_gaussians(output / "gaussian_fields_3d.png", xyz, field_arrays, scales)
        black = torch.zeros(3, device=device)
        rgb_background = torch.ones(3, device=device) if dataset.white_background else black
        encoded = {name: encode_field(value) for name, value in fields.items()}
        names = list(fields)
        splits = ("test", "train") if args.split == "all" else (args.split,)
        view_records = []
        for split in splits:
            views = scene.getTestCameras() if split == "test" else scene.getTrainCameras()
            if args.max_views:
                selected = np.linspace(0, len(views)-1, min(args.max_views, len(views)), dtype=int) if views else []
                views = [views[int(i)] for i in selected]
            for index, view in enumerate(views):
                view.load2device(device)
                deformation = compute_geo_fields(view, gaussians, geo, pipeline, rgb_background,
                                                 opt, iteration, lambda _: 0.,
                                                 1. / max(len(scene.getTrainCameras()), 1), device,
                                                 is_blender=True,
                                                 feature_set="thermal" if view.is_rgbt else "rgb", eval_mode=True)
                render_kwargs = dict(is_6dof=dataset.is_6dof,
                                     feature_set="thermal" if view.is_rgbt else "rgb",
                                     opacity_threshold=float(getattr(opt, "ir_eval_opacity_threshold", 0.0)))

                def raster(colors=None, background=black):
                    return render(view, gaussians, pipeline, background, *deformation, device,
                                  override_color=colors, **render_kwargs)["render"]

                accumulation = raster(torch.ones((n, 3), device=device))[0]
                arrays = {"accumulation": accumulation.cpu().numpy()}
                for start in range(0, len(names), 3):
                    batch = names[start:start+3]
                    channels = [encoded[name][0] for name in batch]
                    channels += [torch.zeros(n, device=device)] * (3-len(channels))
                    packed = raster(torch.stack(channels, dim=1).contiguous())
                    for channel, name in enumerate(batch):
                        _, low, span = encoded[name]
                        decoded, valid = decode_field(packed[channel], accumulation, low, span, args.min_accumulation)
                        arrays[name] = decoded.cpu().numpy()
                mask = valid.cpu().numpy()
                arrays["valid_mask"] = mask.astype(np.uint8)
                if not mask.any():
                    raise ValueError("No visible pixels at the requested accumulation threshold")
                balance_error = float(np.nanmax(np.abs(arrays["q_phys_proxy"] - arrays["self_emission_proxy"] - arrays["reflected_ambient_proxy"] - arrays["residual_proxy"])))
                if balance_error > 2e-5:
                    raise RuntimeError("Rendered radiance decomposition failed: {}".format(balance_error))
                constant_error = None
                if float(fields["ambient_proxy"].max() - fields["ambient_proxy"].min()) < 1e-7:
                    constant_error = float(np.nanmax(np.abs(arrays["ambient_proxy"] - float(fields["ambient_proxy"][0]))))
                    if constant_error > 2e-6:
                        raise RuntimeError("Background normalization failed for constant ambient")
                safe_name = Path(view.image_name).stem
                folder = output / split / ("{:03d}_{}".format(index, safe_name))
                folder.mkdir(parents=True)
                np.savez_compressed(str(folder / "fields.npz"), **arrays)
                for name in list(fields) + ["accumulation"]:
                    scalar_png(arrays[name], folder / (name + ".png"), name, scales)
                Image.fromarray(mask.astype(np.uint8)*255).save(str(folder / "valid_mask.png"))
                tir_gt = rgb_array(view.original_image)
                rgb_gt = rgb_array(view.original_rgb_image) if view.original_rgb_image is not None else tir_gt
                tir_render = rgb_array(raster(background=rgb_background))
                for name, image in (("input_rgb", rgb_gt), ("input_thermal", tir_gt), ("baseline_thermal_render", tir_render)):
                    Image.fromarray(np.rint(image*255).astype(np.uint8)).save(str(folder / (name + ".png")))
                overview(folder / "overview.png", safe_name, arrays, rgb_gt, tir_gt, tir_render, scales, iteration, probe_mode)
                record = {"split": split, "image_name": view.image_name, "directory": str(folder.relative_to(output)),
                          "height": int(view.image_height), "width": int(view.image_width),
                          "valid_pixel_fraction": float(mask.mean()), "radiance_balance_max_error": balance_error,
                          "constant_ambient_max_error": constant_error,
                          "fields": {name: stats(value) for name, value in arrays.items() if name != "valid_mask"}}
                view_records.append(record)
                print("[PROBE_MAPS] {} | valid={:.1%} | balance_error={:.2g}".format(folder.name, mask.mean(), balance_error), flush=True)
                view.load2device("cpu")

    if not view_records:
        raise ValueError("No cameras in the selected split")
    hashes_after = {str(path): digest(path) for path in source_files}
    if hashes_before != hashes_after:
        raise RuntimeError("Source checkpoint assets changed during export")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT)).decode().strip()
    metadata = {"created_utc": datetime.now(timezone.utc).isoformat(), "git_commit": commit,
                "command": " ".join(shlex.quote(v) for v in sys.argv), "iteration": iteration,
                "model_path": str(model_path), "source_path": dataset.source_path, "point_count": n,
                "probe_mode": probe_mode, "probe_metadata": probe_metadata,
                "probe_refitted_during_export": False,
                "temperature_interpretation": "Normalized Planck-LUT Probe estimate, not calibrated absolute surface temperature",
                "temperature_limits_K": [opt.phys_probe_temp_min, opt.phys_probe_temp_max],
                "wavelength_limits_um": [opt.phys_probe_lambda_min_um, opt.phys_probe_lambda_max_um],
                "map_definition": "sum(w_i * attribute_i) / sum(w_i); zero background; affine encoding for signed fields",
                "min_accumulation": args.min_accumulation, "invalid_pixels": "NaN in NPZ, gray in scalar PNGs",
                "coordinates": "Canonical COLMAP coordinates, arbitrary scale; view maps use matching geometry checkpoint",
                "fit_abs_error_definition": "Visibility mean of per-Gaussian absolute fit error; not temperature error or sensor calibration error",
                "q_target_alignment_max_error": q_alignment_error, "temperature_lut_max_error_K": lut_error,
                "checkpoint_sha256_before": hashes_before, "checkpoint_sha256_after": hashes_after,
                "code_sha256": {str(path.relative_to(ROOT)): digest(path) for path in
                                (Path(__file__), ROOT / "utils/probe_maps.py", ROOT / "utils/radiative_utils.py", ROOT / "utils/emissivity_models.py", ROOT / "gaussian_renderer/__init__.py")},
                "display_scales": scales, "gaussian_statistics": {name: stats(value) for name, value in field_arrays.items()},
                "views": view_records}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    (output / "README.md").write_text(
        "# PhysIR-Splat physical maps\n\n"
        "Reused iteration {} with {} Gaussians; no training or Probe refit.\n\n".format(iteration, n)
        + "Temperature is a normalized Planck-LUT Probe estimate (K), **not radiometrically calibrated ground truth**. "
        "All radiance-like fields are dimensionless proxies, not W/(m^2 sr). "
        "Probe mode and coefficients are recorded in metadata.json. No uncertainty is estimated.\n\n"
        "Each view has fields.npz (float32 values; invalid pixels NaN), scalar PNG previews, "
        "a valid mask, input images, a baseline thermal render and overview.png with color bars. "
        "The opacity threshold is a visibility mask, not physical confidence. Shared color scales are in metadata.json.\n\n"
        "self_emission_proxy + reflected_ambient_proxy + residual_proxy = q_phys_proxy. "
        "fit_abs_error_proxy is the visibility mean of pointwise absolute Probe fit errors. "
        "It is not a temperature-error map. Averaged attributes do not equal an independently measured camera temperature.\n\n"
        "gaussian_fields.npz retains every canonical Gaussian center and its attributes; "
        "gaussian_fields_3d.png is a sampled preview in arbitrary COLMAP units.\n\n"
        "metadata.json records matched checkpoint hashes, code hashes, physical settings, validation errors and per-view statistics.\n")
    print("[PROBE_MAPS] DONE: {} views -> {}".format(len(view_records), output), flush=True)


if __name__ == "__main__":
    run()
