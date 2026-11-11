"""Process-local four-material adapter. Main three-class entry points stay unchanged."""
import json
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from utils.material_config import file_sha256
from utils.material_partition import depth_regions, candidate_pairs
import utils.material_config as _config
import utils.material_partition as _partition
ROOT = Path(__file__).resolve().parents[1]
LABELS = dict(unknown=0, nonmetal=1, metal=2, treated_metal=3)
CHANNELS = ('metal', 'nonmetal', 'treated_metal')
_legacy_load = _config.load_material_config


def load_material_config(path=None):
    path = Path(path or _config.DEFAULT_CONFIG).expanduser()
    path = path if path.is_absolute() else ROOT/path
    data = json.loads(path.read_text())
    if data.get('labels') != LABELS:
        return _legacy_load(path)
    if data.get('schema_version') != 2:
        raise ValueError('Require four-material schema version 2')
    reference = (path.parent/data['base_config']).resolve()
    base, _, base_hash = _legacy_load(reference)
    if base_hash != data['base_config_sha256']:
        raise ValueError('Original material config changed')
    if any(data[k] != base[k] for k in ('fusion', 'optimization')):
        raise ValueError('This controlled test preserves fusion/optimization settings')
    if any(data['emissivity'][k] != base['emissivity'][k] for k in ('unknown','nonmetal','metal')):
        raise ValueError('Existing three-class emissivity settings must stay unchanged')
    for name in LABELS:
        v = data['emissivity'][name]
        if v == 'inherit_probe': continue
        low, initial, high = (float(v[k]) for k in ('min','initial','max'))
        if not all(math.isfinite(x) for x in (low,initial,high)) or not 0 < low < initial < high <= 1:
            raise ValueError('Invalid emissivity interval: '+name)
    mask = Path(data['sam3']['mask_directory'])
    if mask.is_absolute() or '..' in mask.parts:
        raise ValueError('Mask directory must remain inside scene')
    return data,path.resolve(),file_sha256(path)


def material_mask_config_hashes(config, path, digest):
    return {digest}


def resolve_material_inputs(scene, opt):
    config,path,digest = load_material_config(opt.phys_material_config)
    if config['labels'] != LABELS:
        raise ValueError('Four-class continuation requires a four-class configuration')
    source=Path(scene.source_path).resolve()
    supplied=getattr(opt,'phys_material_manifest','')
    manifest_path=Path(supplied).resolve() if supplied else source/config['sam3']['mask_directory']/'manifest.json'
    document=json.loads(manifest_path.read_text())
    if (document.get('schema_version') != 2 or document.get('split') != 'train'
            or Path(document.get('scene','')).resolve() != source
            or document.get('labels') != LABELS or not document.get('completed')
            or document.get('config_sha256') != digest):
        raise ValueError('Four-class mask provenance mismatch: '+str(manifest_path))
    return config,path,digest,document,manifest_path


def _frame_data(source,directory,record,height,width,device,image_name):
    image_path=Path(record['image'])
    if (image_path.is_absolute() or '..' in image_path.parts or image_path.parts[:2] != ('rgb','train')
            or image_path.stem != image_name):
        raise ValueError('Require matching RGB TRAIN camera')
    if file_sha256(source/image_path) != record['image_sha256']:
        raise ValueError('RGB source changed')
    paths=[directory/record[k] for k in ('labels','scores')]
    for key,path in zip(('labels','scores'),paths):
        if directory.resolve() not in path.resolve().parents or file_sha256(path) != record[key+'_sha256']:
            raise ValueError('Invalid mask artifact: '+str(path))
    labels=np.asarray(Image.open(paths[0]))
    with np.load(paths[1],allow_pickle=False) as saved:
        scores=np.stack([saved[k] for k in CHANNELS]).astype(np.float32)
    shape=(record['height'],record['width'])
    if (labels.shape != shape or scores.shape != (3,*shape) or not np.isin(labels,list(LABELS.values())).all()
            or not np.isfinite(scores).all() or scores.min()<0 or scores.max()>1):
        raise ValueError('Invalid four-class masks/scores')
    for i,name in enumerate(CHANNELS):
        if np.any(scores[i][labels != LABELS[name]]):
            raise ValueError('Evidence leaks outside its selected class')
    labels=F.interpolate(torch.tensor(labels.astype(np.float32),device=device)[None,None],size=(height,width),mode='nearest')[0,0]
    scores=F.interpolate(torch.tensor(scores,device=device)[None],size=(height,width),mode='bilinear',align_corners=False)[0]
    return labels.cpu().numpy().astype(np.uint8),scores


def classify_evidence(evidence,coverage,observations,options):
    known=evidence.sum(1)
    agreement=evidence.max(1)/np.maximum(known,1e-12)
    confident=((observations>=options['min_views']) &
        (known/np.maximum(coverage,1e-12)>=options['min_known_fraction']) &
        (agreement>=options['min_class_agreement']))
    classes=np.where(confident,np.array([2,1,3])[evidence.argmax(1)],0).astype(np.int64)
    confidence=np.where(confident,evidence.max(1)/np.maximum(coverage,1e-12),0).clip(0,1)
    return classes,confidence.astype(np.float32)


def visibility_adjoint(render,view,gaussians,context,background,deformation,device,kwargs,maps,n):
    blocks=[]
    with torch.enable_grad():
        for start in range(0,len(maps),3):
            count=min(3,len(maps)-start)
            packed=torch.zeros(3,*maps.shape[1:],device=device,dtype=maps.dtype)
            packed[:count]=maps[start:start+count]
            colors=torch.full((n,3),.5,device=device,requires_grad=True)
            image=render(view,gaussians,context['pipeline'],background,*deformation,device,
                         override_color=colors,**kwargs)['render']
            weights=torch.autograd.grad((image*packed).sum()/(view.image_width*view.image_height),colors)[0]
            blocks.append(weights[:,:count].detach())
    return torch.cat(blocks,dim=1)


def install():
    # Only this independently launched Python process sees these adapters.
    _config.load_material_config=load_material_config
    _partition.load_material_config=load_material_config
    _partition.resolve_material_inputs=resolve_material_inputs
    _partition._frame_data=_frame_data
    _partition.classify_evidence=classify_evidence
    _partition.lift_material_partition=lift_material_partition


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
    evidence, coverage, observations = np.zeros((n, 3)), np.zeros(n), np.zeros(n, np.int32)
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
            # Three material channels + coverage require two RGB VJPs.
            # Zero-pad the last pass; never squeeze four maps into RGB.
            weights = visibility_adjoint(render, view, gaussians, context, background,
                                         deformation, device, kwargs, maps, n)
            array = weights.clamp_min(0).cpu().numpy()
            evidence += array[:, :3]
            coverage += array[:, 3]
            observations += (array[:, :3].sum(axis=1) > np.maximum(array[:, 3]*0.05, 1e-10))
            with torch.no_grad():
                clip = homogeneous @ view.full_proj_transform
                ndc = clip[:, :2] / clip[:, 3:4].clamp_min(1e-8)
                pixel = ((ndc+1)*torch.tensor([view.image_width, view.image_height], device=device)/2-0.5).round().long()
                visible = ((clip[:, 3] > 0) & (z > view.znear) & (pixel[:, 0] >= 0) & (pixel[:, 0] < view.image_width) &
                           (pixel[:, 1] >= 0) & (pixel[:, 1] < view.image_height)).cpu().numpy() & (array[:, 3] > 1e-10)
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
