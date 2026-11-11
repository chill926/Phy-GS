#!/usr/bin/env python3
"""One SAM3 image encoding per training view, with direct four-class masks."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.material_four import CHANNELS, LABELS, load_material_config
from tools.segment_sam3_materials import atomic_json
from utils.material_config import file_sha256

COLORS = np.array([(90, 90, 90), (60, 200, 90), (30, 110, 255), (255, 156, 44)], np.uint8)


def segment(processor, image, settings):
    state = processor.set_image(image)
    shape = (image.height, image.width)
    labels = np.zeros(shape, np.uint8)
    scores = {name: np.zeros(shape, np.float32) for name in CHANNELS}
    counts = {}
    for group in sorted(settings["groups"], key=lambda item: item["priority"]):
        name = group["name"]
        evidence = np.zeros(shape, np.float32)
        for prompt in group["prompts"]:
            processor.reset_all_prompts(state)
            result = processor.set_text_prompt(state=state, prompt=prompt)
            counts[prompt] = result["scores"].numel()
            if counts[prompt]:
                probabilities = result["masks_logits"].float().squeeze(1)
                supported = probabilities >= settings["mask_probability_threshold"]
                mask_areas = supported.flatten(1).sum(1)
                accepted = (mask_areas >= 8) & (mask_areas <= probabilities.shape[-2] * probabilities.shape[-1] * .9)
                weight = result["scores"].float() * accepted
                evidence = np.maximum(evidence, (probabilities * supported * weight[:, None, None]).amax(0).cpu().numpy())
        selected = evidence >= settings["pixel_threshold"]
        labels[selected] = LABELS[name]
        scores[name] = evidence
    return labels, {name: np.where(labels == LABELS[name], values, 0).astype(np.float16)
                    for name, values in scores.items()}, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    args = parser.parse_args()
    scene = args.scene.resolve()
    config, config_path, digest = load_material_config(ROOT / "data/materials/sam3_truck_four_fast_v1.json")
    settings = config["sam3"]
    output = scene / settings["mask_directory"]
    images = sorted((scene / "rgb/train").glob("*.jpg"))
    if not images:
        raise ValueError(f"No training JPG files: {scene}")
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = output / "manifest.json"
        identity = dict(schema_version=2, scene=str(scene), split="train", labels=LABELS,
                        config_path=str(config_path), config_sha256=digest,
                        generator_sha256=file_sha256(__file__),
                        checkpoint_sha256=file_sha256(settings["checkpoint"]))
        doc = json.loads(manifest.read_text()) if manifest.exists() else {**identity, "frames": {}}
        if any(doc.get(k) != v for k, v in identity.items()) or not set(doc["frames"]) <= {p.stem for p in images}:
            raise ValueError(f"Incompatible existing masks: {manifest}")
        pending = []
        for path in images:
            record = doc["frames"].get(path.stem)
            if record is None:
                pending.append(path)
            elif file_sha256(path) != record["image_sha256"] or any(
                file_sha256(output / record[k]) != record[k + "_sha256"] for k in ("labels", "scores")
            ):
                raise ValueError(f"Changed input or masks for {path.stem}")
        if pending:
            doc["completed"] = False
            atomic_json(manifest, doc)
            os.environ["HF_HUB_OFFLINE"] = "1"
            from sam3 import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
            torch.set_num_threads(4)
            model = build_sam3_image_model(checkpoint_path=settings["checkpoint"], load_from_HF=False, compile=False)
            processor = Sam3Processor(model, confidence_threshold=settings["instance_threshold"])
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                for path in pending:
                    start = time.monotonic()
                    image = Image.open(path).convert("RGB")
                    labels, scores, counts = segment(processor, image, settings)
                    stem = path.stem
                    label_path, score_path = output / f"{stem}.png", output / f"{stem}.npz"
                    Image.fromarray(labels).save(label_path)
                    np.savez_compressed(score_path, **scores)
                    Image.fromarray(COLORS[labels]).save(output / f"{stem}.labels_rgb.png")
                    overlay = (0.60 * np.asarray(image) + 0.40 * COLORS[labels]).astype(np.uint8)
                    Image.fromarray(overlay).save(output / f"{stem}.overlay.jpg", quality=93)
                    totals = {name: int((labels == index).sum()) for name, index in LABELS.items()}
                    seconds = time.monotonic() - start
                    doc["frames"][stem] = dict(image=str(path.relative_to(scene)), image_sha256=file_sha256(path),
                        width=image.width, height=image.height, labels=label_path.name, scores=score_path.name,
                        labels_sha256=file_sha256(label_path), scores_sha256=file_sha256(score_path),
                        pixel_counts=totals, detections=counts, seconds=seconds, image_encodes=1)
                    atomic_json(manifest, doc)
                    print(f"[{len(doc['frames'])}/{len(images)}] {stem} {seconds:.2f}s {totals}", flush=True)
        doc["completed"] = True
        atomic_json(manifest, doc)
        print(f"Truck: {len(images)} verified four-class training masks in {output}", flush=True)


if __name__ == "__main__":
    main()
