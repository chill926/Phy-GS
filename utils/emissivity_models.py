"""Constrained, temperature-dependent band-effective emissivity."""

import math
import torch


def linear_emissivity_settings(opt):
    low, high = float(opt.phys_probe_e_min), float(opt.phys_probe_e_max)
    t_min, t_max = float(opt.phys_probe_temp_min), float(opt.phys_probe_temp_max)
    t_ref = float(getattr(opt, "phys_probe_temp_ref", 300.0))
    slope = float(getattr(opt, "phys_probe_emissivity_slope", 0.0))
    initial = float(opt.phys_probe_e_init)
    if not all(math.isfinite(v) for v in (low, high, t_min, t_max, t_ref, slope, initial)):
        raise ValueError("Linear emissivity settings must be finite")
    if not (0.0 < low < high <= 1.0 and 0.0 < t_min < t_max):
        raise ValueError("Require 0 < e_min < e_max <= 1 and 0 < temp_min < temp_max")
    if not t_min <= t_ref <= t_max:
        raise ValueError("phys_probe_temp_ref must lie within the fitted temperature interval")
    offsets = (slope * (t_min-t_ref), slope * (t_max-t_ref))
    # Endpoint constraints suffice for an affine function over the whole interval.
    eps0_low, eps0_high = low-min(offsets), high-max(offsets)
    if eps0_high-eps0_low <= 1e-7:
        raise ValueError("Emissivity slope leaves no feasible eps0 interval over the temperature range")
    if not eps0_low < initial < eps0_high:
        raise ValueError("phys_probe_e_init must lie inside feasible eps0 interval ({:.8g}, {:.8g}); "
                         "reduce the slope or supply a compatible reference emissivity".format(eps0_low, eps0_high))
    return {"eps0_min": eps0_low, "eps0_max": eps0_high,
            "temperature_ref_K": t_ref, "slope_per_K": slope}


def linear_emissivity(eps0_cluster, temperature, material_ids, settings):
    return eps0_cluster[material_ids] + settings["slope_per_K"] * (temperature-settings["temperature_ref_K"])


def learnable_emissivity_settings(opt):
    settings = linear_emissivity_settings(opt)
    if settings["slope_per_K"] != 0.0:
        raise ValueError("learnable_linear_emissivity starts at zero slope; set phys_probe_emissivity_slope=0")
    weight = float(getattr(opt, "phys_probe_lambda_slope", 0.001))
    warmup = float(getattr(opt, "phys_probe_slope_warmup_fraction", 0.25))
    lr_scale = float(getattr(opt, "phys_probe_slope_lr_scale", 0.25))
    if not all(math.isfinite(v) for v in (weight, warmup, lr_scale)):
        raise ValueError("Learnable slope settings must be finite")
    if weight < 0.0 or not 0.0 <= warmup < 1.0 or lr_scale <= 0.0:
        raise ValueError("Require lambda_slope >= 0, 0 <= slope_warmup_fraction < 1, slope_lr_scale > 0")
    settings.update(emissivity_min=float(opt.phys_probe_e_min),
                    emissivity_max=float(opt.phys_probe_e_max),
                    temperature_min_K=float(opt.phys_probe_temp_min),
                    temperature_max_K=float(opt.phys_probe_temp_max),
                    slope_regularization_weight=weight,
                    slope_warmup_fraction=warmup, slope_lr_scale=lr_scale)
    return settings


def endpoint_emissivity_parameters(center_raw, difference_raw, settings, freeze_slope=False):
    """Two endpoint logits written as a shared center and their difference.

    difference_raw=0 gives exactly constant emissivity. Detaching the difference
    during warmup still lets the common center (reference emissivity) learn.
    Both endpoints lie in bounds, hence so does their affine interpolation.
    """
    difference = difference_raw.detach() if freeze_slope else difference_raw
    low, high = settings["emissivity_min"], settings["emissivity_max"]
    e_low = low + (high-low) * torch.sigmoid(center_raw - 0.5*difference)
    e_high = low + (high-low) * torch.sigmoid(center_raw + 0.5*difference)
    span = settings["temperature_max_K"] - settings["temperature_min_K"]
    ref_fraction = (settings["temperature_ref_K"] - settings["temperature_min_K"]) / span
    eps0 = (1.0-ref_fraction)*e_low + ref_fraction*e_high
    slope = (e_high-e_low) / span
    return eps0, slope, e_low, e_high


def probe_model_metadata(opt):
    mode = getattr(opt, "phys_probe_mode", "original")
    if mode not in ("original", "linear_emissivity", "learnable_linear_emissivity"):
        raise ValueError("Unknown phys_probe_mode: {}".format(mode))
    metadata = {"schema_version": 2, "mode": mode, "temperature_model": "normalized_planck_lut",
                "temperature_min_K": float(opt.phys_probe_temp_min),
                "temperature_max_K": float(opt.phys_probe_temp_max),
                "wavelength_min_um": float(opt.phys_probe_lambda_min_um),
                "wavelength_max_um": float(opt.phys_probe_lambda_max_um),
                "temperature_ref_K": float(getattr(opt, "phys_probe_temp_ref", 300.0)),
                "temperature_is_radiometrically_calibrated": False,
                "eps0_parameterization": "fitted_per_material_cluster_with_prior",
                "slope_per_K": 0.0, "slope_parameterization": "zero"}
    if mode == "linear_emissivity":
        metadata.update(linear_emissivity_settings(opt))
        metadata["slope_parameterization"] = "fixed_scalar_per_K_shared_across_clusters"
    elif mode == "learnable_linear_emissivity":
        settings = learnable_emissivity_settings(opt)
        metadata.update(settings)
        metadata.update(schema_version=3, slope_per_K=None,
                        initial_slope_per_K=settings["slope_per_K"],
                        eps0_parameterization="derived_from_bounded_endpoint_emissivities",
                        slope_parameterization="learned_per_material_cluster_bounded_endpoint_logits",
                        slope_regularization="mean_square_endpoint_difference_divided_by_emissivity_span",
                        warmup_scope="inner_steps_of_each_independent_probe_fit")
    return metadata
