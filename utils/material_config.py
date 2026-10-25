"""Shared material configuration; importable in both sam3 and 3dgs environments."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "data/materials/sam3_materials.json"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_material_config(path=None):
    path = Path(path or DEFAULT_CONFIG).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    data = json.loads(path.read_text())
    if data.get("schema_version") != 1 or data.get("labels") != {"unknown": 0, "nonmetal": 1, "metal": 2}:
        raise ValueError("Unsupported material schema or label IDs")
    for name, settings in data["emissivity"].items():
        if settings == "inherit_probe":
            continue
        low, initial, high = (float(settings[key]) for key in ("min", "initial", "max"))
        if not all(math.isfinite(x) for x in (low, initial, high)) or not 0 < low < initial < high <= 1:
            raise ValueError("Require 0 < min < initial < max <= 1 for " + name)
    for options in [data["sam3"]["prompts"], *data["sam3"].get("scene_prompts", {}).values()]:
        for name in ("metal", "nonmetal"):
            if not options.get(name) or not all(isinstance(p, str) and p.strip() for p in options[name]):
                raise ValueError("Each class needs non-empty text prompts")
    for key in ("instance_threshold", "pixel_threshold", "conflict_margin"):
        if not 0 <= data["sam3"][key] <= 1:
            raise ValueError("Invalid SAM3 threshold: " + key)
    sam = data['sam3']
    directory = Path(sam.get('mask_directory', 'material_masks/sam3'))
    if directory.is_absolute() or '..' in directory.parts:
        raise ValueError('Material mask directory must remain inside its scene')
    if sam.get('pipeline_version', 1) not in (1, 2):
        raise ValueError('Unsupported material segmentation pipeline')
    if sam.get('pipeline_version', 1) == 2:
        if not 0 < sam['mask_probability_threshold'] < 1:
            raise ValueError('Invalid mask probability threshold')
        for groups in [sam['groups'], *sam.get('scene_groups', {}).values()]:
            if not groups or len({g['name'] for g in groups}) != len(groups):
                raise ValueError('Material groups must have unique names')
            for group in groups:
                if (group['class'] not in ('metal', 'nonmetal') or not group['prompts'] or
                        not all(isinstance(p, str) and p.strip() for p in group['prompts']) or
                        not isinstance(group['priority'], int) or not 0 <= group['priority'] < 32768 or
                        not 0 < group['instance_threshold'] <= 1 or not group['views'] or
                        not set(group['views']) <= {'full', 'anchor_base'}):
                    raise ValueError('Invalid SAM3 material group: '+group['name'])
                for key in ('max_mask_fraction', 'max_crop_mask_fraction'):
                    if not 0 < group.get(key, 1) <= 1:
                        raise ValueError('Invalid instance area filter: '+key)
        crop = sam['anchor_crops']
        if (crop['max_anchors'] < 1 or crop['min_size'] < 16 or crop['width_factor'] <= 0 or
                crop['height_factor'] <= 0 or not 0 <= crop['bottom_padding'] < 1):
            raise ValueError('Invalid object-relative crop settings')
        if (not all(-.5 <= value <= .5 for value in crop.get('retry_horizontal_offsets', [])) or
                crop.get('context_scale', 1.25) < 1 or not 0 <= crop.get('context_bottom_padding', .35) < 1):
            raise ValueError('Invalid crop retry settings')
    f = data["fusion"]
    if not 1 <= f["min_views"] <= f["max_views"] or f["neighbors"] < 1:
        raise ValueError("Invalid fusion view count or neighbor count")
    for key in ("min_known_fraction", "min_class_agreement", "min_accumulation", "max_region_conflict_fraction"):
        if not 0 < f[key] <= 1:
            raise ValueError("Invalid fusion fraction: " + key)
    if f["relative_depth_jump"] <= 0 or f["min_region_pixels"] < 1 or f["neighbor_distance_factor"] <= 0 or f["min_same_region_views"] < 1:
        raise ValueError("Invalid depth/region settings")
    stage = data["optimization"]
    if not 0 <= stage["emissivity_start_fraction"] <= stage["slope_start_fraction"] < 1 or stage["emissivity_lr_scale"] <= 0:
        raise ValueError("Invalid material optimization stages")
    return data, path.resolve(), file_sha256(path)


def material_mask_config_hashes(config, config_path, config_hash):
    """Accept an old mask config only if its actual segmentation inputs match.

    A reference preserves the provenance of existing masks when changing only
    Probe/fusion settings. Never rewrite or relabel an old mask manifest.
    """
    accepted = {config_hash}
    reference = config.get("mask_source_config")
    if reference:
        path = Path(reference).expanduser()
        if not path.is_absolute():
            path = Path(config_path).parent / path
        previous, _, previous_hash = load_material_config(path)
        if any(previous[key] != config[key] for key in ("schema_version", "labels", "sam3")):
            raise ValueError("mask_source_config has different SAM3 segmentation settings; regenerate masks")
        accepted.add(previous_hash)
    return accepted


def material_bounds(config, opt, cluster_classes, device):
    import torch
    low = torch.full(cluster_classes.shape, float(opt.phys_probe_e_min), device=device)
    high = torch.full_like(low, float(opt.phys_probe_e_max))
    initial = torch.full_like(low, float(opt.phys_probe_e_init))
    for name, label in config["labels"].items():
        settings = config["emissivity"][name]
        if settings != "inherit_probe":
            selected = cluster_classes == label
            low[selected], high[selected], initial[selected] = settings["min"], settings["max"], settings["initial"]
    override = float(getattr(opt, "phys_metal_e_init", -1.0))
    half_range = float(getattr(opt, "phys_metal_e_half_range", -1.0))
    if override != -1.0:
        settings = config["emissivity"]["metal"]
        if half_range != -1.0:
            minimum, maximum = override-half_range, override+half_range
            if not all(math.isfinite(v) for v in (override, half_range)) or not 0 <= minimum < override < maximum <= 1:
                raise ValueError("Metal override requires 0 <= center-half_range < center < center+half_range <= 1")
        else:
            minimum, maximum = settings["min"], settings["max"]
            if not math.isfinite(override) or not minimum < override < maximum:
                raise ValueError("Metal initial override must be strictly inside configured bounds")
        selected = cluster_classes == config["labels"]["metal"]
        low[selected], high[selected], initial[selected] = minimum, maximum, override
    elif half_range != -1.0:
        raise ValueError("Metal range override requires an explicit metal initial value")
    if not ((low >= 0) & (low < initial) & (initial < high) & (high <= 1)).all():
        raise ValueError("Initial emissivity must be strictly inside each material interval")
    return low, high, initial
