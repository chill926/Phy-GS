#!/usr/bin/env python3
"""Matched continuation: original training versus freezing only base Gaussian scales.

Runs the existing train.training function without editing production files.
Both arms load the same weights with fresh Adam and evaluate before any update.
The IR/RGB deformation networks, SH, XYZ, rotations and opacity remain trainable.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from tools.diagnose_geometry import load_config, digest, write_json
from scene import GaussianModel, Scene
import train


class Tee:
    def __init__(self, terminal, log):
        self.terminal, self.log = terminal, log

    def write(self, text):
        self.terminal.write(text)
        self.log.write(text)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        return False


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=["control", "freeze_scale"], required=True)
    parser.add_argument("--start", type=int, default=20000)
    parser.add_argument("--end", type=int, default=30000)
    args = parser.parse_args()
    os.chdir(ROOT)
    source, out = Path(args.model_path).resolve(), Path(args.output).resolve()
    if not 0 < args.start < args.end or out == source or source in out.parents:
        parser.error("Require 0 < start < end and a new output outside the source run")
    if out.exists():
        parser.error("Output already exists; use a new directory")
    device = torch.device("cuda:0")
    cfg = load_config(source, device)
    if args.start < cfg.densify_until_iter:
        parser.error("Start must be at/after densification ends; point indices must remain fixed")
    cfg.model_path = str(out)
    cfg.load_model_path = str(source)
    cfg.load_iteration = args.start
    cfg.iterations = args.end
    cfg.resume_eval_before_train = True
    cfg.resume_refit_probe = False
    out.mkdir(parents=True)
    logfile = open(out / "console.log", "w", buffering=1)
    sys.stdout, sys.stderr = Tee(sys.stdout, logfile), Tee(sys.stderr, logfile)
    protected = list(ROOT.glob("*.py")) + list((ROOT / "scene").glob("*.py"))
    protected += list((ROOT / "utils").glob("*.py")) + [ROOT / "arguments/__init__.py", ROOT / "gaussian_renderer/__init__.py"]
    protected += list((ROOT / "data/materials").glob("*.json"))
    protected += [source / "cfg_args", source / "result.txt"]
    protected += [p for directory in ("point_cloud", "Geometry", "GeometryRGB", "Atmos", "Planck", "Radiometric")
                  for p in (source / directory / f"iteration_{args.start}").glob("*") if p.is_file()]
    before = {str(p): digest(p) for p in protected}
    metadata = {"status": "running", "mode": args.mode, "source": str(source), "output": str(out),
                "start": args.start, "end": args.end, "seed": 0, "fresh_Adam": True,
                "only_intervention": "base _scaling.requires_grad_(False)" if args.mode == "freeze_scale" else "none",
                "deformation_scale_still_trainable": True, "protected_before": before}
    write_json(out / "scale_check.json", metadata)
    camera_log = open(out / "camera_order.jsonl", "w", buffering=1)
    old_setup, old_scene, old_randint = GaussianModel.training_setup, Scene.__init__, train.randint
    state = {}

    def setup(model, options, model_device):
        old_setup(model, options, model_device)
        state["model"] = model
        metadata["initial_gaussian_sha256"] = {
            name: tensor_hash(getattr(model, name)) for name in (
                "_xyz", "_features_dc", "_features_rest", "_thermal_features_dc", "_thermal_features_rest",
                "_scaling", "_rotation", "_opacity")}
        state["initial_scale"] = model._scaling.detach().clone()
        if args.mode == "freeze_scale":
            model._scaling.requires_grad_(False)
        metadata["base_scale_requires_grad"] = model._scaling.requires_grad
        write_json(out / "scale_check.json", metadata)
        print(f"[SCALE-CHECK] mode={args.mode} base_scale_requires_grad={model._scaling.requires_grad}", flush=True)

    def scene_init(scene, *positional, **keywords):
        old_scene(scene, *positional, **keywords)
        state["cameras"] = [camera.image_name for camera in scene.getTrainCameras()]
        metadata["initial_camera_order"] = state["cameras"]
        write_json(out / "scale_check.json", metadata)

    def pick(a, b):
        chosen = old_randint(a, b)
        if not state.get("remaining"):
            state["remaining"] = state["cameras"].copy()
        name = state["remaining"].pop(chosen)
        step = args.start + state.get("picks", 0) + 1
        state["picks"] = state.get("picks", 0) + 1
        camera_log.write(json.dumps({"iteration": step, "view": name}) + "\n")
        return chosen

    GaussianModel.training_setup, Scene.__init__, train.randint = setup, scene_init, pick
    started = time.monotonic()
    try:
        train.safe_state(False, device)
        train.training(cfg, cfg, cfg, list(range(args.start, args.end + 1, 1000)),
                       [args.start, args.end], device)
        model = state["model"]
        delta = (model._scaling.detach() - state["initial_scale"]).abs()
        metadata["final_base_scale_sha256"] = tensor_hash(model._scaling)
        metadata["base_log_scale_max_abs_change"] = float(delta.max())
        metadata["base_log_scale_mean_abs_change"] = float(delta.mean())
        metadata["camera_picks"] = state["picks"]
        if args.mode == "freeze_scale" and not torch.equal(model._scaling.detach(), state["initial_scale"]):
            raise RuntimeError("Frozen base scales changed")
        if state["picks"] != args.end - args.start:
            raise RuntimeError("Unexpected continuation step count")
        metadata["status"] = "complete"
    except BaseException as exc:
        metadata.update(status="failed", error=repr(exc))
        raise
    finally:
        metadata["elapsed_seconds"] = time.monotonic() - started
        metadata["protected_after"] = {str(p): digest(p) for p in protected}
        metadata["source_assets_unchanged"] = metadata["protected_after"] == before
        if not metadata["source_assets_unchanged"]:
            metadata["status"] = "validation_failed"
        write_json(out / "scale_check.json", metadata)
        GaussianModel.training_setup, Scene.__init__, train.randint = old_setup, old_scene, old_randint
        camera_log.close()
    print(f"[SCALE-CHECK] {metadata['status'].upper()}: {out}", flush=True)
    if metadata["status"] != "complete":
        raise RuntimeError("Scale check validation failed")


if __name__ == "__main__":
    main()
