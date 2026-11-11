#!/usr/bin/env python3
"""Regenerate the historical RoadBlock or Truck SAM3 training-view masks."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
MATERIALS = ROOT / "data/materials"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(tool, *args):
    subprocess.run([sys.executable, "-u", str(ROOT / "tools" / tool),
                    *map(str, args)], cwd=ROOT, check=True)


def validate_frames(document, scene, folder, frames):
    if (document["completed"] is not True or document["split"] != "train"
            or Path(document["scene"]) != scene or set(document["frames"]) != frames):
        raise ValueError(f"Incomplete or mismatched train masks: {folder}")
    for name, frame in document["frames"].items():
        for key, directory in (("image", scene), ("labels", folder), ("scores", folder)):
            path = directory / frame[key]
            if directory not in path.resolve().parents and key != "image":
                raise ValueError(f"Mask path escapes its output: {path}")
            if sha(path) != frame[key + "_sha256"]:
                raise ValueError(f"Corrupt {key} for {name}: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", choices=("RoadBlock", "Truck"))
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "data/RGBT-Scenes")
    args = parser.parse_args()
    scene = (args.dataset_root / args.scene).resolve()
    frames = {p.stem for p in (scene / "rgb/train").glob("*.jpg")}
    if not frames:
        raise ValueError(f"No RGB training JPG files: {scene}")
    if args.scene == "Truck":
        return run("segment_sam3_truck_four_fast.py", "--scene", scene)

    masks = scene / "material_masks"
    masks.mkdir(exist_ok=True)
    with (masks / ".sam3-segmentation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = MATERIALS / "sam3_materials_eps020_reference.json"
        output = masks / "sam3_v2"
        run("segment_sam3_materials.py", "--scene", scene, "--config", config, "--resume")
        document = json.loads((output / "manifest.json").read_text())
        validate_frames(document, scene, output, frames)
        if document["config_sha256"] != sha(config):
            raise ValueError(f"Wrong configuration for {output}")
        print(f"RoadBlock: {len(frames)} verified training masks in {output}", flush=True)


if __name__ == "__main__":
    main()
