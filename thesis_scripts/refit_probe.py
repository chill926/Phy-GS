#!/usr/bin/env python3
"""Refit only the Probe and create a separate, renderable experiment directory.

No 3DGS optimizer is run. Point cloud and geometry assets are linked read-only
by convention to the existing experiment; the original experiment must remain.
"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from argparse import ArgumentParser, Namespace
import hashlib
import json
import shlex
import torch

from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from scene import Scene, GaussianModel, GeoRefineModel
from utils.emissivity_models import probe_model_metadata
from utils.radiative_utils import fit_and_log_physir_attributes


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def main():
    device = torch.device("cuda:0")
    parser = ArgumentParser(description=__doc__)
    model_group = ModelParams(parser, device, sentinel=True)
    pipe_group = PipelineParams(parser)
    parser.set_defaults(**{name: None for name in vars(pipe_group)})
    opt_group = OptimizationParams(parser, sentinel=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--output-model", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = get_combined_args(parser, device)
    defaults = vars(OptimizationParams(ArgumentParser())).copy()
    defaults.update(vars(args))
    args = Namespace(**defaults)
    opt = opt_group.extract(args)
    model_metadata = probe_model_metadata(opt)  # Validate before creating output.
    source = Path(args.model_path).resolve()
    destination = Path(args.output_model).resolve()
    if destination.exists():
        raise FileExistsError("Choose a new --output-model; refusing to overwrite " + str(destination))
    point = source / "point_cloud" / ("iteration_{}".format(args.iteration)) / "point_cloud.ply"
    assets = [point, source / "cfg_args"]
    for folder in ("Geometry", "GeometryRGB"):
        weight = source / folder / ("iteration_{}".format(args.iteration)) / "geometry.pth"
        if weight.exists():
            assets.append(weight)
    if not point.is_file():
        raise FileNotFoundError(str(point))
    hashes = {str(p): sha256(p) for p in assets}
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    dataset = model_group.extract(args)
    dataset.model_path = str(source)
    dataset.load_model_path = ""
    dataset.data_device = device
    dataset.load2gpu_on_the_fly = True
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, device)
        scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    from utils.material_partition import resolve_material_inputs
    material_context = None
    if resolve_material_inputs(scene, opt) is not None:
        from train import compute_geo_fields
        pipeline = pipe_group.extract(args)
        geo = GeoRefineModel(device, dataset.is_blender, dataset.is_6dof,
                             time_multires=opt.geometry_time_multires if opt.geometry_time_multires >= 0 else None)
        subdir = "GeometryRGB" if scene.has_rgbt else "Geometry"
        geo.load_weights(str(source), iteration=args.iteration, subdir=subdir)
        geo.geometry.eval()
        background = torch.ones(3, device=device) if dataset.white_background else torch.zeros(3, device=device)
        def material_deformation(view):
            return compute_geo_fields(view, gaussians, geo, pipeline, background, opt, args.iteration,
                                      lambda _: 0., 1./max(len(scene.getTrainCameras()), 1), device,
                                      dataset.is_blender, feature_set="rgb", eval_mode=True)
        material_context = {"pipeline": pipeline, "deformation": material_deformation, "is_6dof": dataset.is_6dof}
    destination.mkdir(parents=True)
    checkpoint = destination / "point_cloud" / ("iteration_{}".format(args.iteration))
    checkpoint.mkdir(parents=True)
    (checkpoint / "point_cloud.ply").symlink_to(point)
    for folder in ("Geometry", "GeometryRGB", "Atmos", "Planck", "Radiometric"):
        if (source / folder).exists():
            (destination / folder).symlink_to(source / folder, target_is_directory=True)
    n = int(gaussians.get_xyz.shape[0])
    summary = fit_and_log_physir_attributes(args.iteration, gaussians, scene, opt, str(destination), n,
                                          material_context=material_context)
    if not gaussians.save_radiative_transfer_attributes(str(checkpoint / "phys_probe.pt")):
        raise RuntimeError("Probe cache was not saved")
    saved = dict(vars(args))
    saved.update(model_path=str(destination), source_path=dataset.source_path,
                 load_model_path="", load_iteration=0, data_device=device)
    (destination / "cfg_args").write_text(str(Namespace(**saved)))
    after = {str(p): sha256(p) for p in assets}
    if hashes != after:
        raise RuntimeError("Source assets changed during the refit")
    manifest = {"source_model": str(source), "output_model": str(destination),
                "iteration": args.iteration, "seed": args.seed, "point_count": n,
                "mode": model_metadata, "source_sha256_before": hashes,
                "source_sha256_after": after, "source_assets_unchanged": True,
                "command": " ".join(shlex.quote(v) for v in sys.argv),
                "summary": summary}
    (destination / "refit_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("[REFIT_PROBE] Done: {}".format(destination), flush=True)


if __name__ == "__main__":
    main()
