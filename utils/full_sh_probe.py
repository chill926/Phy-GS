"""Detached, train-view full-SH observations for the material Probe.

The observation unit is display gray, not DN, radiance or temperature. Keep
directional samples: their weighted mean is only an auxiliary statistic.
"""
from contextlib import contextmanager
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from utils.sh_utils import SH2RGB, eval_sh


MODE = "full_sh_observations"
ATTRIBUTE_MODE = "full_sh_radiometric_attributes"
MODES = (MODE, ATTRIBUTE_MODE)
DOMAIN = "display_gray_proxy_NOT_DN_NOT_physical_radiance"


def configure_full_sh_probe(opt):
    """An observation-only Probe must not consume old physical-field feedback."""
    if getattr(opt, "phys_probe_mode", "") not in MODES:
        return
    if getattr(opt, "ir_temporal_affine", False) or getattr(opt, "ir_temporal_pose", False):
        raise ValueError("Full-SH observations currently require no temporal affine/pose correction")
    views = int(opt.phys_probe_sh_views)
    minimum = float(opt.phys_probe_sh_min_acc)
    tolerance = float(opt.phys_probe_sh_reference_tolerance)
    if views < 0 or not math.isfinite(minimum) or not 0 < minimum <= 1:
        raise ValueError("Invalid full-SH view count or accumulation threshold")
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Invalid full-SH renderer reference tolerance")
    if opt.phys_probe_mode == MODE:
        opt.phys_aux_mse_max_weight = 0.0
    opt.phys_guidance_max_weight = 0.0
    opt.phys_render_max_blend = 0.0
    opt.phys_main_render = False
    opt.phys_eval_with_render = False
    stage = ("live epsilon(T) attributes with GT-MSE feedback" if opt.phys_probe_mode == ATTRIBUTE_MODE
             else "no temperature/emissivity inversion")
    print("[FULL-SH PROBE] detached train-view observations; "+stage, flush=True)


@contextmanager
def _preserve_rng(device):
    # Material fallback clustering uses random samples. Diagnostics must not
    # change the subsequent training camera order or DropGaussian realization.
    py_state, np_state = random.getstate(), np.random.get_state()
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)


def _save_tensor_state(path, state):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def _json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def _stats(values):
    if not values.numel():
        return None
    values = values.detach().double()
    return {"mean": float(values.mean()), "p05_p50_p95": torch.quantile(
        values, values.new_tensor([.05, .5, .95])).cpu().tolist()}


def _material_partition(scene, gaussians, opt, context):
    from utils.material_partition import lift_material_partition
    from utils.radiative_utils import _material_features, _assign_kmeans
    features = gaussians.get_thermal_features.detach()
    # These features retain the established SAM3 unknown-region fallback;
    # thermal DC is not used as a Probe target or inverted to temperature.
    auxiliary = {"rgb_dc": SH2RGB(features[:, 0]).clamp(0, 1),
                 "rest_ratio": features[:, 1:].reshape(len(features), -1).norm(dim=1, keepdim=True)
                 / features[:, 0].norm(dim=1, keepdim=True).clamp_min(1e-8)}
    clustering_features = _material_features(gaussians, auxiliary, opt)
    def fallback(selected):
        return _assign_kmeans(clustering_features[selected], int(opt.phys_decouple_clusters),
                              int(opt.phys_decouple_cluster_samples), int(opt.phys_decouple_cluster_iters))
    return lift_material_partition(scene, gaussians, opt, context, fallback)


def run_full_sh_probe(iteration, gaussians, scene, opt, model_path, ir_context,
                      material_context=None):
    """Save all selected TRAIN directions and alpha-composited contributions.

    The primary render/geometry parameters are detached. Only dummy colors
    receive an autograd VJP; existing main-optimizer gradients stay untouched.
    Visibility uses the loaded training image grid, without a raw-DN mapping.
    """
    device = gaussians.get_xyz.device
    with _preserve_rng(device):
        return _run(iteration, gaussians, scene, opt, model_path, ir_context, material_context)


def _run(iteration, g, scene, opt, model_path, context, material_context):
    from gaussian_renderer import render
    from utils.rigid_utils import from_homogenous, to_homogenous
    start = time.monotonic()
    n = len(g.get_xyz)
    if n == 0:
        return None
    device = g.get_xyz.device
    pipe = context["pipeline"]
    is_6dof = context["is_6dof"]
    background = context["background"].detach()
    black = torch.zeros_like(background)
    train = sorted(scene.getTrainCameras(), key=lambda v: v.image_name)
    test_names = {v.image_name for v in scene.getTestCameras()}
    if not train or len({v.image_name for v in train}) != len(train):
        raise ValueError("Full-SH Probe requires unique nonempty training views")
    if test_names.intersection(v.image_name for v in train):
        raise ValueError("Train/test names overlap; refusing to build Probe observations")
    maximum = int(opt.phys_probe_sh_views)
    views = train if maximum == 0 or maximum >= len(train) else [
        train[i] for i in np.linspace(0, len(train)-1, maximum, dtype=int)]
    with torch.no_grad():
        features = g.get_thermal_features.detach()
        if features.shape != (n, (g.max_sh_degree+1)**2, 3):
            raise ValueError("Unexpected thermal SH coefficient layout")
        degree = int(g.active_sh_degree)
        gray = features.new_tensor([.299, .587, .114])
        partition = _material_partition(scene, g, opt, material_context)

    weights_sum = torch.zeros(n, device=device, dtype=torch.float64)
    weights_squared = torch.zeros_like(weights_sum)
    mean, m2 = torch.zeros_like(weights_sum), torch.zeros_like(weights_sum)
    counts = torch.zeros(n, device=device, dtype=torch.int32)
    values_cpu, weights_cpu, directions_cpu, records = [], [], [], []
    minimum = float(opt.phys_probe_sh_min_acc)
    tolerance = float(opt.phys_probe_sh_reference_tolerance)
    print("[FULL-SH PROBE][iter {}] {} train views, {} points, active degree {}".format(
        iteration, len(views), n, degree), flush=True)
    for index, view in enumerate(views):
        old_device = view.world_view_transform.device
        view.load2device(device)
        try:
            with torch.no_grad():
                deformation = tuple(x.detach() if torch.is_tensor(x) else x
                                    for x in context["deformation"](view))
                xyz = g.get_xyz.detach()
                if pipe.convert_SHs_python:
                    means = xyz  # Match the current renderer's Python path.
                elif is_6dof and torch.is_tensor(deformation[0]):
                    means = from_homogenous(torch.bmm(deformation[0], to_homogenous(xyz).unsqueeze(-1)).squeeze(-1))
                else:
                    means = xyz + deformation[0]
                directions = F.normalize(means-view.camera_center[None], dim=1)
                rgb = (eval_sh(degree, features.transpose(1, 2), directions)+.5).clamp_min(0)
                scalar = rgb @ gray
                if not torch.isfinite(scalar).all():
                    raise FloatingPointError("Non-finite full-SH observations")

                def raster(colors=None, bg=background):
                    return render(view, g, pipe, bg, *deformation, device,
                                  is_6dof=is_6dof, feature_set="thermal", override_color=colors,
                                  detach_geometry=True, disable_atmosphere=True)["render"]
                reference = raster()
                full = raster(rgb.contiguous())
                scalar_image = raster(scalar[:, None].expand(-1, 3).contiguous())
                reference_error = float((reference-full).abs().max())
                projection_error = float((scalar_image[0]-(full*gray[:, None, None]).sum(0)).abs().max())
                if not math.isfinite(reference_error+projection_error) or max(reference_error, projection_error) > tolerance:
                    raise RuntimeError("Full-SH renderer mismatch at {}: {} / {}".format(
                        view.image_name, reference_error, projection_error))
                acc = raster(torch.ones(n, 3, device=device), black)[0]
                valid = torch.isfinite(acc) & (acc >= minimum)
            with torch.enable_grad():
                dummy = torch.zeros(n, 3, device=device, requires_grad=True)
                image = raster(dummy, black)[0]
                weight = torch.autograd.grad((image*valid).sum(), dummy)[0][:, 0].detach().clamp_min(0)
            with torch.no_grad():
                if not torch.isfinite(weight).all():
                    raise FloatingPointError("Non-finite Probe contribution weights")
                wd, yd = weight.double(), scalar.double()
                new_sum = weights_sum+wd
                delta = yd-mean
                new_mean = mean+wd/new_sum.clamp_min(1e-30)*delta
                m2 += wd*delta*(yd-new_mean)
                mean, weights_sum = new_mean, new_sum
                weights_squared += wd.square()
                counts += (weight > 1e-7).int()
                values_cpu.append(scalar.cpu())
                weights_cpu.append(weight.cpu())
                directions_cpu.append(directions.cpu())
                records.append({"view": view.image_name, "valid_pixels": int(valid.sum()),
                                "image_size": [view.image_width, view.image_height],
                                "contributing_points": int((weight > 1e-7).sum()),
                                "full_sh_render_max_abs_error": reference_error,
                                "scalar_projection_max_abs_error": projection_error})
        finally:
            view.load2device(old_device)
        if index == 0 or (index+1) % 8 == 0 or index+1 == len(views):
            print("[FULL-SH PROBE] {}/{} {} | reference {:.3g} gray {:.3g}".format(
                index+1, len(views), view.image_name, reference_error, projection_error), flush=True)

    with torch.no_grad():
        supported = weights_sum > 1e-7
        std = (m2/weights_sum.clamp_min(1e-30)).clamp_min(0).sqrt()
        effective = weights_sum.square()/weights_squared.clamp_min(1e-30)
        labels = torch.zeros(n, device=device, dtype=torch.long) if partition is None else partition["point_classes"]
        state = {"kind": "full_sh_probe_input_observations", "schema_version": 1,
                 "scalar_domain": DOMAIN, "physical_temperature_estimated": False,
                 "emissivity_estimated": False, "main_loss_feedback": False,
                 "source_model": str(Path(model_path).resolve()), "source_iteration": int(iteration),
                 "active_sh_degree": degree, "max_sh_degree": g.max_sh_degree,
                 "thermal_sh_coefficients": features.cpu(), "base_xyz": g.get_xyz.detach().cpu(),
                 "fit_view_names": [v.image_name for v in views], "training_views_only": True,
                 "visibility_grid": "loaded_training_image_grid_without_DN_mapping",
                 "min_accumulation": minimum, "full_opacity_no_dropout": True,
                 "direction_convention": "base_xyz-camera_center for Python SH; refined means-camera_center for CUDA SH",
                 "fit_full_SH_scalar": torch.stack(values_cpu),
                 "fit_visibility_weights": torch.stack(weights_cpu),
                 "fit_SH_directions": torch.stack(directions_cpu),
                 "full_SH_weighted_mean": mean.cpu(), "full_SH_weighted_std": std.cpu(),
                 "supported": supported.cpu(), "contributing_view_count": counts.cpu(),
                 "effective_view_count": effective.cpu(),
                 "legacy_DC_scalar": (SH2RGB(features[:, 0]).clamp(0, 1) @ gray).cpu(),
                 "material_class": labels.cpu(), "view_records": records}
        if partition is not None:
            from utils.material_config import material_bounds
            low, high, initial = material_bounds(partition["config"], opt, partition["cluster_classes"], device)
            state.update(material_ids=partition["material_ids"].cpu(),
                         material_confidence=partition["point_confidence"].cpu(),
                         material_partition=partition["metadata"],
                         cluster_class=partition["cluster_classes"].cpu(),
                         cluster_emissivity_prior=initial.cpu(),
                         cluster_emissivity_prior_bounds=torch.stack([low, high], dim=-1).cpu())
        summary = {"iteration": int(iteration), "probe_mode": MODE, "scalar_domain": DOMAIN,
                   "active_sh_degree": degree, "coefficients_per_gaussian": features.shape[1]*3,
                   "points": n, "supported_points": int(supported.sum()), "train_views": len(views),
                   "physical_temperature_estimated": False, "emissivity_estimated": False,
                   "main_loss_feedback": False, "visibility_grid": state["visibility_grid"],
                   "max_full_sh_render_error": max(r["full_sh_render_max_abs_error"] for r in records),
                   "max_scalar_projection_error": max(r["scalar_projection_max_abs_error"] for r in records),
                   "point_statistics": {name: {"points": int(mask.sum()),
                       "view_std": _stats(std[mask]), "effective_views": _stats(effective[mask])}
                       for name, mask in [("all", supported), ("metal", supported & (labels == 2)),
                                          ("nonmetal", supported & (labels == 1)), ("unknown", supported & (labels == 0))]}}
    folder = Path(model_path)/"phys_probe"/("iteration_%d" % iteration)
    folder.mkdir(parents=True, exist_ok=True)
    observation_path = folder/"full_sh_observations.pt"
    _save_tensor_state(observation_path, state)
    summary.update(observations_path=str(observation_path), elapsed_seconds=time.monotonic()-start)
    _json(folder/"summary.json", summary)
    with (folder.parent/"probe_log.jsonl").open("a") as stream:
        stream.write(json.dumps(summary, ensure_ascii=False, allow_nan=False)+"\n")
    line = "[FULL-SH PROBE][iter {}] saved {} views, {}/{} supported | {:.1f}s | observation cache only".format(
        iteration, len(views), int(supported.sum()), n, summary["elapsed_seconds"])
    print(line, flush=True)
    with (Path(model_path)/"result.txt").open("a") as stream:
        stream.write(line+"\n")
    return summary
