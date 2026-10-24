"""Numerical helpers for visibility-normalized physical attribute maps.

These maps visualize cached Probe attributes; they are not calibrated camera
temperatures. Keeping these helpers independent of the scene loader also makes
the signed-field and background normalization rules testable on CPU.
"""

import torch


def build_probe_fields(state, point_count):
    names = ("temperature", "emissivity", "emission", "ambient", "delta", "q_phys", "q_target")
    values = {}
    for name in names:
        value = state[name].detach().reshape(-1).float()
        if value.numel() != point_count or not torch.isfinite(value).all():
            raise ValueError("Invalid Probe field {} for {} points".format(name, point_count))
        values[name] = value
    e, u, a = values["emissivity"], values["emission"], values["ambient"]
    expected = e * u + (1.0 - e) * a + values["delta"]
    if not torch.allclose(expected, values["q_phys"], atol=2e-6, rtol=1e-5):
        raise ValueError("Probe does not satisfy q_phys = epsilon*u + (1-epsilon)*ambient + delta")
    fields = {
        "temperature_K": values["temperature"],
        "emissivity": e,
        "emission_proxy": u,
        "ambient_proxy": a,
        "self_emission_proxy": e * u,
        "reflected_ambient_proxy": (1.0 - e) * a,
        "residual_proxy": values["delta"],
        "q_phys_proxy": values["q_phys"],
        "q_target_proxy": values["q_target"],
        "fit_abs_error_proxy": (values["q_phys"] - values["q_target"]).abs(),
    }
    for name in ("emissivity_ref", "emissivity_slope"):
        if name in state and state[name].numel():
            value = state[name].detach().reshape(-1).float()
            if value.numel() != point_count or not torch.isfinite(value).all():
                raise ValueError("Invalid optional Probe field: " + name)
            fields[name] = value
    metadata = state.get("probe_metadata", {})
    if metadata.get("mode") in ("linear_emissivity", "learnable_linear_emissivity"):
        if not all(name in fields for name in ("emissivity_ref", "emissivity_slope")):
            raise ValueError("Linear Probe cache is missing its reference emissivity or slope")
        predicted = fields["emissivity_ref"] + fields["emissivity_slope"] * (fields["temperature_K"] - metadata["temperature_ref_K"])
        if not torch.allclose(predicted, fields["emissivity"], atol=2e-6, rtol=1e-5):
            raise ValueError("Cached emissivity does not match epsilon(T)")
    return fields


def encode_field(value):
    """Affine encoding avoids signed colors and image-range clipping."""
    low = float(value.min().item())
    high = float(value.max().item())
    span = high - low if high > low else 1.0
    return (value - low) / span, low, span


def decode_field(numerator, accumulation, low, span, min_accumulation):
    valid = torch.isfinite(accumulation) & (accumulation >= min_accumulation)
    value = low + span * numerator / accumulation.clamp_min(1e-8)
    valid = valid & torch.isfinite(value)
    return torch.where(valid, value, torch.full_like(value, float("nan"))), valid
