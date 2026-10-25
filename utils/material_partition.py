"""Lift train-view SAM3 evidence with rasterizer visibility, then depth-aware grouping."""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from utils.material_config import file_sha256, load_material_config, material_mask_config_hashes


def depth_regions(labels, depth, accumulation, options):
    """Connected image regions: equal material and small relative neighboring depth jump."""
    labels = np.asarray(labels)
    depth = np.asarray(depth)
    valid = (labels > 0) & np.isfinite(depth) & (depth > 0) & (accumulation >= options["min_accumulation"])
    height, width = labels.shape
    pixels = np.arange(height*width).reshape(height, width)
    sources, targets = [], []
    for a, b in ((np.s_[:, :-1], np.s_[:, 1:]), (np.s_[:-1, :], np.s_[1:, :])):
        jump = np.abs(depth[a]-depth[b]) / np.maximum(np.minimum(depth[a], depth[b]), 1e-8)
        connected = valid[a] & valid[b] & (labels[a] == labels[b]) & (jump <= options["relative_depth_jump"])
        sources.append(pixels[a][connected])
        targets.append(pixels[b][connected])
    source, target = np.concatenate(sources), np.concatenate(targets)
    graph = coo_matrix((np.ones(len(source), np.uint8), (source, target)), shape=(height*width, height*width))
    _, component = connected_components(graph, directed=False)
    sizes = np.bincount(component[valid.reshape(-1)], minlength=height*width)
    keep = valid.reshape(-1) & (sizes[component] >= options["min_region_pixels"])
    return np.where(keep, component+1, 0).reshape(height, width).astype(np.int32)


def candidate_pairs(xyz, options):
    if len(xyz) < 2:
        return np.empty((0, 2), np.int64)
    distance, neighbor = cKDTree(xyz).query(xyz, k=min(len(xyz), options["neighbors"]+1), workers=4)
    nearest = np.where(np.isfinite(distance) & (distance > 1e-10), distance, np.inf).min(axis=1)
    positive = nearest[np.isfinite(nearest)]
    radius = (np.median(positive) if len(positive) else 1e-3) * options["neighbor_distance_factor"]
    a = np.broadcast_to(np.arange(len(xyz))[:, None], neighbor.shape)
    valid = (a < neighbor) & np.isfinite(distance) & (distance <= radius)
    return np.stack((a[valid], neighbor[valid]), axis=1)


def classify_evidence(evidence, coverage, observations, options):
    """SAM3 scores are evidence, not calibrated material probabilities."""
    known = evidence.sum(axis=1)
    agreement = evidence.max(axis=1) / np.maximum(known, 1e-12)
    confident = ((observations >= options["min_views"]) &
                 (known / np.maximum(coverage, 1e-12) >= options["min_known_fraction"]) &
                 (agreement >= options["min_class_agreement"]))
    classes = np.where(confident, np.where(evidence[:, 0] > evidence[:, 1], 2, 1), 0).astype(np.int64)
    confidence = np.where(confident, evidence.max(axis=1) / np.maximum(coverage, 1e-12), 0).clip(0, 1)
    return classes, confidence.astype(np.float32)


def resolve_material_inputs(scene, opt):
    mode = getattr(opt, "phys_material_mode", "auto")
    if mode not in ("auto", "legacy", "sam3_depth"):
        raise ValueError("Unknown phys_material_mode: " + mode)
    if mode == "legacy":
        return None
    source = Path(scene.source_path).resolve()
    supplied = getattr(opt, "phys_material_manifest", "")
    config, config_path, config_hash = load_material_config(getattr(opt, "phys_material_config", None))
    mask_directory = config['sam3'].get('mask_directory', 'material_masks/sam3')
    path = Path(supplied).expanduser().resolve() if supplied else source / mask_directory / 'manifest.json'
    if not path.is_file():
        if mode == "sam3_depth" or supplied:
            raise FileNotFoundError("Generate SAM3 train masks before enabling material partition: " + str(path))
        return None
    manifest = json.loads(path.read_text())
    accepted_hashes = material_mask_config_hashes(config, config_path, config_hash)
    if (manifest.get("schema_version") != 1 or manifest.get("split") != "train" or
            Path(manifest.get("scene", "")).resolve() != source or manifest.get("config_sha256") not in accepted_hashes or
            manifest.get("labels") != config["labels"] or not manifest.get("completed")):
        raise ValueError("Material masks are incomplete or do not match this scene/configuration: " + str(path))
    return config, config_path, config_hash, manifest, path


def _frame_data(source, directory, record, height, width, device, image_name):
    image_path = Path(record["image"])
    if (image_path.is_absolute() or ".." in image_path.parts or image_path.parts[:2] != ("rgb", "train") or
            image_path.stem != image_name):
        raise ValueError("Material cache image does not match its RGB training camera")
    paths = [directory / record[key] for key in ("labels", "scores")]
    if any(directory.resolve() not in path.resolve().parents for path in paths):
        raise ValueError("Material cache path escapes its manifest directory")
    if file_sha256(source / record["image"]) != record["image_sha256"]:
        raise ValueError("RGB source changed after SAM3 segmentation")
    for key, path in zip(("labels", "scores"), paths):
        if file_sha256(path) != record[key+"_sha256"]:
            raise ValueError("Corrupt material cache: " + str(path))
    labels = np.asarray(Image.open(paths[0]))
    with np.load(paths[1], allow_pickle=False) as saved:
        scores = np.stack((saved["metal"], saved["nonmetal"])).astype(np.float32)
    expected = (record["height"], record["width"])
    if labels.shape != expected or scores.shape != (2, *expected) or not np.isin(labels, [0, 1, 2]).all():
        raise ValueError("Invalid material cache shape or label IDs")
    if not np.isfinite(scores).all() or scores.min() < 0 or scores.max() > 1:
        raise ValueError("Invalid material evidence scores")
    labels = F.interpolate(torch.tensor(labels.astype(np.float32), device=device)[None, None], size=(height, width), mode="nearest")[0, 0]
    scores = F.interpolate(torch.tensor(scores, device=device)[None], size=(height, width), mode="bilinear", align_corners=False)[0]
    return labels.cpu().numpy().astype(np.uint8), scores


def lift_material_partition(scene, gaussians, opt, context, fallback_cluster):
    inputs = resolve_material_inputs(scene, opt)
    if inputs is None:
        return None
    if context is None:
        raise ValueError("SAM3 material partition requires the matching RGB render/geometry context")
    from gaussian_renderer import render
    from utils.rigid_utils import from_homogenous, to_homogenous

    config, config_path, config_hash, manifest, manifest_path = inputs
    options = config["fusion"]
    all_views = sorted(scene.getTrainCameras(), key=lambda view: view.image_name)
    views = [view for view in all_views if view.image_name in manifest["frames"]]
    if not views:
        raise ValueError("No material masks correspond to training cameras")
    # A pilot manifest is useful for explicit refit validation, but auto training needs complete train coverage.
    if getattr(opt, "phys_material_mode", "auto") == "auto" and len(views) != len(all_views):
        raise ValueError("Auto mode requires masks for every training view; generate the full mask set")
    if len(views) > options["max_views"]:
        views = [views[i] for i in np.linspace(0, len(views)-1, options["max_views"], dtype=int)]
    if len(views) < options["min_views"]:
        raise ValueError("Too few material training views for multi-view fusion")
    xyz = gaussians.get_xyz.detach()
    device, n = xyz.device, len(xyz)
    pairs = candidate_pairs(xyz.cpu().numpy(), options)
    same, conflict = np.zeros(len(pairs), np.int32), np.zeros(len(pairs), np.int32)
    evidence, coverage, observations = np.zeros((n, 2)), np.zeros(n), np.zeros(n, np.int32)
    background = torch.zeros(3, device=device)
    records = []
    for view in views:
        old_device = view.world_view_transform.device
        view.load2device(device)
        try:
            with torch.no_grad():
                deformation = context["deformation"](view)
                d_xyz = deformation[0]
                if context["is_6dof"] and torch.is_tensor(d_xyz):
                    positions = from_homogenous(torch.bmm(d_xyz, to_homogenous(xyz).unsqueeze(-1)).squeeze(-1))
                else:
                    positions = xyz + d_xyz
                homogeneous = torch.cat((positions, torch.ones((n, 1), device=device)), dim=1)
                z = (homogeneous @ view.world_view_transform)[:, 2]
                packed = torch.stack((z.clamp_min(0), torch.ones_like(z), torch.ones_like(z)), dim=1)
                kwargs = dict(is_6dof=context["is_6dof"], feature_set="rgb", detach_geometry=True)
                depth_render = render(view, gaussians, context["pipeline"], background, *deformation, device,
                                      override_color=packed.contiguous(), **kwargs)["render"]
                accumulation = depth_render[1].detach()
                depth = (depth_render[0] / accumulation.clamp_min(1e-8)).detach()
                labels, scores = _frame_data(Path(scene.source_path), manifest_path.parent,
                                            manifest["frames"][view.image_name], view.image_height, view.image_width, device, view.image_name)
                regions = depth_regions(labels, depth.cpu().numpy(), accumulation.cpu().numpy(), options)
                valid_pixels = torch.tensor(regions > 0, device=device)
                scores = scores * valid_pixels
                geometry_valid = (accumulation >= options["min_accumulation"]) & torch.isfinite(depth) & (depth > 0)
                maps = torch.cat((scores, geometry_valid[None].float()), dim=0)
            with torch.enable_grad():
                # The color adjoint is the exact alpha/transmittance-weighted contribution of each Gaussian.
                colors = torch.full((n, 3), 0.5, device=device, requires_grad=True)
                image = render(view, gaussians, context["pipeline"], background, *deformation, device,
                               override_color=colors, **kwargs)["render"]
                weights = torch.autograd.grad((image*maps).sum() / (view.image_width*view.image_height), colors)[0].detach()
            array = weights.clamp_min(0).cpu().numpy()
            evidence += array[:, :2]
            coverage += array[:, 2]
            observations += (array[:, :2].sum(axis=1) > np.maximum(array[:, 2]*0.05, 1e-10))
            with torch.no_grad():
                clip = homogeneous @ view.full_proj_transform
                ndc = clip[:, :2] / clip[:, 3:4].clamp_min(1e-8)
                pixel = ((ndc+1)*torch.tensor([view.image_width, view.image_height], device=device)/2-0.5).round().long()
                visible = ((clip[:, 3] > 0) & (z > view.znear) & (pixel[:, 0] >= 0) & (pixel[:, 0] < view.image_width) &
                           (pixel[:, 1] >= 0) & (pixel[:, 1] < view.image_height)).cpu().numpy() & (array[:, 2] > 1e-10)
                pixel = pixel.cpu().numpy()
            region_ids = np.zeros(n, np.int32)
            region_ids[visible] = regions[pixel[visible, 1], pixel[visible, 0]]
            if len(pairs):
                a, b = region_ids[pairs[:, 0]], region_ids[pairs[:, 1]]
                eligible = (a > 0) & (b > 0)
                same += eligible & (a == b)
                conflict += eligible & (a != b)
            records.append({"image_name": view.image_name, "depth_regions": int(len(np.unique(regions[regions > 0]))),
                            "visible_gaussians": int(visible.sum())})
        finally:
            view.load2device(old_device)
    classes, confidence = classify_evidence(evidence, coverage, observations, options)
    a, b = pairs[:, 0], pairs[:, 1]
    keep = ((classes[a] > 0) & (classes[a] == classes[b]) & (same >= options["min_same_region_views"]) &
            (conflict / np.maximum(same+conflict, 1) <= options["max_region_conflict_fraction"]))
    graph = coo_matrix((np.ones(int(keep.sum()), np.uint8), (a[keep], b[keep])), shape=(n, n))
    _, components = connected_components(graph, directed=False)
    recognized = classes > 0
    ids = torch.empty(n, device=device, dtype=torch.long)
    cluster_classes = []
    if recognized.any():
        _, first, inverse = np.unique(components[recognized], return_index=True, return_inverse=True)
        ids[torch.tensor(recognized, device=device)] = torch.tensor(inverse, device=device)
        cluster_classes = classes[recognized][first].tolist()
    if (~recognized).any():
        selected = torch.tensor(~recognized, device=device)
        fallback = fallback_cluster(selected)
        ids[selected] = fallback + len(cluster_classes)
        cluster_classes.extend([0] * (int(fallback.max())+1))
    metadata = {"source": "sam3_train_rgb_alpha_adjoint_and_depth_regions", "config_path": str(config_path),
                "config_sha256": config_hash, "manifest": str(manifest_path), "manifest_sha256": file_sha256(manifest_path),
                "checkpoint_sha256": manifest["checkpoint_sha256"], "train_views": records,
                "point_counts": {name: int((classes == label).sum()) for name, label in config["labels"].items()},
                "clusters": len(cluster_classes), "depth_graph_edges": int(keep.sum()),
                "class_labels": config["labels"], "unknown_policy": "legacy_probe_bounds_and_prior",
                "refresh_policy": "recomputed_for_current_gaussians_on_each_probe; no point-index cache"}
    print("[MATERIAL] {} clusters={} depth_edges={}".format(metadata["point_counts"], len(cluster_classes), int(keep.sum())), flush=True)
    return {"material_ids": ids, "cluster_classes": torch.tensor(cluster_classes, device=device, dtype=torch.long),
            "point_classes": torch.tensor(classes, device=device, dtype=torch.long),
            "point_confidence": torch.tensor(confidence, device=device), "config": config, "metadata": metadata}
