"""Material-bounded Probe fit with staged release of reference emissivity and slope."""
import math

import torch
import torch.nn.functional as F

from utils.emissivity_models import endpoint_emissivity_parameters
from utils.material_config import material_bounds


def fit_partitioned_attributes(extracted, gaussians, opt, device, lut, partition, smooth_pairs):
    q = extracted["q_target"].detach().reshape(-1).clamp(1e-4, 1-1e-4)
    n = q.numel()
    ids = partition["material_ids"]
    classes = partition["cluster_classes"]
    low, high, initial = material_bounds(partition["config"], opt, classes, device)
    mode = getattr(opt, "phys_probe_mode", "original")
    learn_slope = mode == "learnable_linear_emissivity"
    slope = float(getattr(opt, "phys_probe_emissivity_slope", 0)) if mode == "linear_emissivity" else 0.0
    t_min, t_max, t_ref = opt.phys_probe_temp_min, opt.phys_probe_temp_max, opt.phys_probe_temp_ref
    offsets = (slope*(t_min-t_ref), slope*(t_max-t_ref))
    eps0_low, eps0_high = low-min(offsets), high-max(offsets)
    if not ((eps0_low < initial) & (initial < eps0_high)).all():
        raise ValueError("Fixed emissivity slope leaves no feasible class-specific initialization")
    a0 = float(q.quantile(opt.phys_probe_ambient_quantile).clamp(1e-4, 0.95))
    u_raw = torch.logit(q).clone().requires_grad_(True)
    a_raw = torch.tensor(math.log(a0/(1-a0)), device=device, requires_grad=True)
    delta_raw = torch.zeros(n, device=device, requires_grad=True)
    center = torch.logit(((initial-eps0_low)/(eps0_high-eps0_low)).clamp(1e-4, 1-1e-4)).detach().requires_grad_(True)
    effective_initial = (eps0_low+(eps0_high-eps0_low)*center.detach().sigmoid()).clone()
    difference = torch.zeros_like(center, requires_grad=True)
    stages = partition["config"]["optimization"]
    lr = float(getattr(opt, "phys_decouple_lr", opt.phys_probe_lr))
    steps = max(1, int(getattr(opt, "phys_decouple_steps", opt.phys_probe_steps)))
    eps_start = int(steps*stages["emissivity_start_fraction"])
    slope_start = int(steps*stages["slope_start_fraction"])
    groups = [{"params": [u_raw, a_raw, delta_raw]},
              {"params": [center], "lr": lr*stages["emissivity_lr_scale"]}]
    if learn_slope:
        groups.append({"params": [difference], "lr": lr*opt.phys_probe_slope_lr_scale})
    optimizer = torch.optim.Adam(groups, lr=lr)
    endpoints = {"emissivity_min": low, "emissivity_max": high, "temperature_min_K": t_min,
                 "temperature_max_K": t_max, "temperature_ref_K": t_ref}
    i, j, pair_weights = smooth_pairs
    # Preserve spatial smoothing inside each depth/material region; never couple different fitted regions.
    same_region = ids[i] == ids[j]
    i, j, pair_weights = i[same_region], j[same_region], pair_weights[same_region]
    delta_max = float(getattr(opt, "phys_decouple_delta_max", opt.phys_probe_delta_max))

    def evaluate(step):
        c = center.detach() if step < eps_start else center
        if learn_slope:
            eps0, k, _, _ = endpoint_emissivity_parameters(c, difference, endpoints, freeze_slope=step < slope_start)
        else:
            eps0 = eps0_low+(eps0_high-eps0_low)*c.sigmoid()
            k = torch.full_like(eps0, slope)
        u = u_raw.sigmoid()
        temperature = lut.norm_radiance_to_temp(u)
        e = eps0[ids] + k[ids]*(temperature-t_ref)
        ambient = a_raw.sigmoid()
        delta = delta_max*(delta_raw.detach() if step < eps_start else delta_raw).tanh()
        prediction = e*u+(1-e)*ambient+delta
        return eps0, k, u, temperature, e, ambient, delta, prediction

    for step in range(steps):
        eps0, k, u, temperature, e, ambient, delta, prediction = evaluate(step)
        beta = opt.phys_probe_huber_beta
        loss = F.smooth_l1_loss(prediction, q, beta=beta)
        loss = loss + opt.phys_decouple_lambda_e_prior*F.smooth_l1_loss(eps0, initial, beta=beta)
        loss = loss + opt.phys_decouple_lambda_a_prior*F.smooth_l1_loss(ambient, ambient.new_tensor(a0), beta=beta)
        loss = loss + opt.phys_decouple_lambda_delta*delta.abs().mean()
        if len(i) and opt.phys_decouple_lambda_temp_smooth > 0:
            temp_unit = (temperature-t_min)/(t_max-t_min)
            difference_loss = F.smooth_l1_loss(temp_unit[i], temp_unit[j], beta=opt.phys_decouple_temp_smooth_beta, reduction="none")
            loss = loss + opt.phys_decouple_lambda_temp_smooth*(pair_weights*difference_loss).sum()/pair_weights.sum().clamp_min(1e-6)
        if learn_slope:
            loss = loss + opt.phys_probe_lambda_slope*((k*(t_max-t_min)/(high-low))**2).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite material Probe loss at step {}".format(step))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        eps0, k, u, temperature, e, ambient, delta, prediction = evaluate(steps)
        if not torch.isfinite(e).all() or not ((e >= low[ids]-1e-6) & (e <= high[ids]+1e-6)).all():
            raise FloatingPointError("Material Probe violated its per-class emissivity bounds")
        report = dict(partition["metadata"])
        if float(getattr(opt, "phys_metal_e_init", -1.0)) != -1.0:
            report["runtime_overrides"] = {
                "metal_initial_and_prior": float(opt.phys_metal_e_init),
                "metal_half_range": float(getattr(opt, "phys_metal_e_half_range", -1.0)),
                "base_config_and_segmentation_unchanged": True,
            }
        report["emissivity_by_class"] = {name: {"min": float(low[classes == label][0]), "max": float(high[classes == label][0]),
                                               "initial": float(initial[classes == label][0]),
                                               "effective_initial_eps0": float(effective_initial[classes == label][0])}
                                           for name, label in partition["config"]["labels"].items() if (classes == label).any()}
        report["optimization"] = dict(stages, steps=steps, emissivity_start_step=eps_start, slope_start_step=slope_start,
                                      delta_start_step=eps_start, smoothing="within_same_material_depth_region")
        return {"emission": u[:, None], "emissivity": e[:, None], "ambient": ambient.expand(n, 1),
                "delta": delta[:, None], "q_phys": prediction[:, None], "temperature": temperature[:, None],
                "fit_abs_err": (prediction-q).abs()[:, None], "fit_sq_err": ((prediction-q)**2)[:, None],
                "ambient_init": torch.full((n, 1), a0, device=device), "material_ids": ids[:, None],
                "cluster_emissivity": eps0[:, None], "cluster_eps0": eps0[:, None], "cluster_k": k[:, None],
                "cluster_counts": torch.bincount(ids, minlength=len(classes)).float()[:, None],
                "emissivity_ref": eps0[ids, None], "emissivity_slope": k[ids, None],
                "material_class": partition["point_classes"][:, None], "material_confidence": partition["point_confidence"][:, None],
                "cluster_class": classes[:, None], "emissivity_min": low[ids, None], "emissivity_max": high[ids, None],
                "_material_metadata": report}
