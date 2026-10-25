#!/usr/bin/env python3
"""Generate material masks from RGB TRAIN views using the separate sam3 environment."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from utils.material_config import DEFAULT_CONFIG, file_sha256, load_material_config
from utils.material_segmentation import accept_instance, anchor_crop, compose_material_instances

PIPELINE_VERSION = 2
COLORS = np.array([[110, 110, 110], [60, 200, 90], [30, 110, 255]], dtype=np.uint8)


def assign_material_pixels(metal, nonmetal, threshold, margin):
    """Legacy flat-score policy, retained for reproducing v1 configurations."""
    labels = np.zeros(metal.shape, dtype=np.uint8)
    labels[(metal >= threshold) & (metal - nonmetal >= margin)] = 2
    labels[(nonmetal >= threshold) & (nonmetal - metal >= margin)] = 1
    return labels


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def configuration_groups(config, scene_name):
    sam = config['sam3']
    if sam.get('pipeline_version', 1) == 1:
        return None
    return sam.get('scene_groups', {}).get(scene_name, sam['groups'])


def segment_image(processor, image, groups, settings):
    shape = (image.height, image.width)
    full = (0, 0, image.width, image.height)
    instances, detections, anchors = [], [], []
    mask_threshold = settings['mask_probability_threshold']

    def infer(window, view_name, only_groups=None):
        state = processor.set_image(image.crop(window))
        for group in groups:
            if view_name not in group['views']:
                continue
            if only_groups is not None and group['name'] not in only_groups:
                continue
            processor.set_confidence_threshold(group['instance_threshold'])
            for prompt in group['prompts']:
                processor.reset_all_prompts(state)
                result = processor.set_text_prompt(state=state, prompt=prompt)
                probabilities = result['masks_logits'].float().squeeze(1).cpu().numpy()
                scores = result['scores'].float().cpu().numpy()
                boxes = result['boxes'].float().cpu().numpy()
                accepted = 0
                for probability, score, box in zip(probabilities, scores, boxes):
                    if not accept_instance(probability, float(score), group, shape, mask_threshold):
                        continue
                    box = (box + np.array([window[0], window[1], window[0], window[1]])).tolist()
                    item = dict(group=group['name'], **{'class': group['class']}, priority=group['priority'],
                                prompt=prompt, score=float(score), box=box, window=window,
                                view=view_name, probabilities=probability,
                                mask_pixels=int((probability >= mask_threshold).sum()))
                    instances.append(item)
                    accepted += 1
                    if view_name == 'full' and group.get('crop_anchor', False):
                        anchors.append(item)
                detections.append(dict(group=group['name'], prompt=prompt, view=view_name, window=window,
                                       detected=len(scores), accepted=accepted))

    infer(full, 'full')
    crops = []
    crop_options = settings.get('anchor_crops', {})
    if crop_options.get('enabled', False) and any('anchor_base' in g['views'] for g in groups):
        for item in sorted(anchors, key=lambda item: item['score'], reverse=True):
            crop = anchor_crop(item['box'], image.size, crop_options)
            if crop not in crops:
                crops.append(crop)
            if len(crops) >= crop_options['max_anchors']:
                break
        for crop in crops:
            infer(crop, 'anchor_base')
        # Retry only configured small-object groups, using object-relative windows.
        # This uses no per-frame coordinates or test images.
        retry_groups = {g['name'] for g in groups if g.get('retry_crops', False)}
        for anchor in sorted(anchors, key=lambda item: item['score'], reverse=True)[:crop_options['max_anchors']]:
            for offset in crop_options.get('retry_horizontal_offsets', []):
                missing = {name for name in retry_groups if not any(item['group'] == name for item in instances)}
                if not missing:
                    break
                crop = anchor_crop(anchor['box'], image.size, dict(crop_options, horizontal_offset=offset))
                if crop not in crops:
                    crops.append(crop)
                    infer(crop, 'anchor_base', missing)
            needs_context = set()
            for name in retry_groups:
                found = [item for item in instances if item['group'] == name]
                if not found or any(any(abs(a-b) < 4 for a,b in zip(item['box'], item['window'])) for item in found):
                    needs_context.add(name)
            if needs_context:
                factor = crop_options.get('context_scale', 1.25)
                options = dict(crop_options, width_factor=crop_options['width_factor']*factor,
                               height_factor=crop_options['height_factor']*factor,
                               bottom_padding=crop_options.get('context_bottom_padding', .35))
                crop = anchor_crop(anchor['box'], image.size, options)
                if crop not in crops:
                    crops.append(crop)
                    infer(crop, 'anchor_base', needs_context)
    labels, metal, nonmetal, priority = compose_material_instances(
        instances, shape, mask_threshold, settings['pixel_threshold'], settings['conflict_margin'])
    records = [{key: value for key, value in item.items() if key != 'probabilities'} for item in instances]
    return labels, metal, nonmetal, dict(instances=records, detections=detections, crops=crops,
                                       conflict_pixels=int(((metal > 0) & (nonmetal > 0) & (labels == 0)).sum()))


def save_previews(output, stem, image, labels, records):
    Image.fromarray(COLORS[labels]).save(output/(stem+'.labels_rgb.png'))
    overlay = (0.6*np.asarray(image)+0.4*COLORS[labels]).clip(0, 255).astype(np.uint8)
    Image.fromarray(overlay).save(output/(stem+'.overlay.jpg'))
    font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font = ImageFont.truetype(font_path, 17) if Path(font_path).exists() else ImageFont.load_default()
    canvas = Image.new('RGB', (2*image.width, image.height+76), 'white')
    canvas.paste(image, (0, 40))
    canvas.paste(Image.fromarray(COLORS[labels]), (image.width, 40))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 10), stem+' | RGB', fill='black', font=font)
    draw.text((image.width+8, 10), 'Pure labels | blue=metal, green=nonmetal, gray=unknown', fill='black', font=font)
    number = 0
    drawn = []
    for item in records.get('instances', []):
        if item['class'] != 'metal':
            continue
        # One outline per distinct metal box, not one for every synonymous prompt.
        box = tuple(int(round(x)) for x in item['box'])
        if any(sum(abs(a-b) for a,b in zip(box, old)) < 12 for old in drawn):
            continue
        drawn.append(box)
        number += 1
        draw.rectangle((box[0],box[1]+40,box[2],box[3]+40),outline=(30,110,255),width=2)
        draw.text((box[2]+2,box[1]+40),str(number),fill=(0,80,240),font=font)
    counts = {name:int((labels==value).sum()) for name,value in [('unknown',0),('nonmetal',1),('metal',2)]}
    draw.text((8,image.height+48),str(counts)+' | semantic labels, not measured material properties',fill='black',font=font)
    canvas.save(output/(stem+'.qa.jpg'),quality=94)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', required=True)
    parser.add_argument('--config', default=str(DEFAULT_CONFIG))
    parser.add_argument('--output')
    parser.add_argument('--max-images',type=int,default=0,help='Train-view pilot only; 0 processes all train RGB images')
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args()
    if args.max_images < 0:
        parser.error('--max-images must be nonnegative')
    config,config_path,config_hash = load_material_config(args.config)
    scene = Path(args.scene).resolve()
    images = sorted(p for p in (scene/'rgb/train').iterdir() if p.suffix.lower() in ('.jpg','.jpeg','.png'))
    if not images or len({p.stem for p in images}) != len(images):
        raise ValueError('Require unique RGB train image names')
    if args.max_images:
        images = [images[i] for i in np.linspace(0,len(images)-1,min(args.max_images,len(images)),dtype=int)]
    output = Path(args.output).resolve() if args.output else scene/config['sam3'].get('mask_directory','material_masks/sam3')
    if output.exists() and not args.resume:
        raise FileExistsError('Use a new directory, or --resume for an unchanged configuration: '+str(output))
    output.mkdir(parents=True,exist_ok=True)
    settings = config['sam3']
    groups = configuration_groups(config,scene.name)
    prompts = settings.get('scene_prompts',{}).get(scene.name,settings['prompts'])
    checkpoint = Path(settings['checkpoint'])
    checkpoint_hash = file_sha256(checkpoint)
    header = dict(schema_version=1,generator_version=PIPELINE_VERSION,scene=str(scene),split='train',
                  generator_sha256=file_sha256(__file__),rules_sha256=file_sha256(ROOT/'utils/material_segmentation.py'),
                  config_sha256=config_hash,config_path=str(config_path),checkpoint_sha256=checkpoint_hash,
                  prompts=groups if groups is not None else prompts,labels=config['labels'],frames={})
    manifest_path = output/'manifest.json'
    if args.resume and manifest_path.exists():
        previous=json.loads(manifest_path.read_text())
        if any(previous.get(k)!=header[k] for k in ('generator_version','generator_sha256','rules_sha256','scene','split','config_sha256','checkpoint_sha256','prompts')):
            raise ValueError('Resume would mix incompatible material masks or generator versions')
        header=previous
    header['completed']=False
    atomic_json(manifest_path,header)
    os.environ['HF_HUB_OFFLINE']='1'
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    torch.set_num_threads(8)
    model=build_sam3_image_model(checkpoint_path=str(checkpoint),load_from_HF=False,compile=False)
    processor=Sam3Processor(model,confidence_threshold=settings['instance_threshold'])
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for index,image_path in enumerate(images):
            stem,image_hash=image_path.stem,file_sha256(image_path)
            if stem in header['frames']:
                old=header['frames'][stem]
                if old['image_sha256']!=image_hash or any(file_sha256(output/old[k])!=old[k+'_sha256'] for k in ('labels','scores')):
                    raise ValueError('Stale/corrupt frame cache: '+stem)
                continue
            start=time.perf_counter()
            image=Image.open(image_path).convert('RGB')
            if groups is not None:
                labels,metal,nonmetal,details=segment_image(processor,image,groups,settings)
            else:
                state=processor.set_image(image)
                processor.set_confidence_threshold(settings['instance_threshold'])
                scores={name:np.zeros((image.height,image.width),np.float32) for name in ('metal','nonmetal')}
                details={'detections':{}}
                for name in ('metal','nonmetal'):
                    for prompt in prompts[name]:
                        processor.reset_all_prompts(state)
                        result=processor.set_text_prompt(state=state,prompt=prompt)
                        details['detections'][prompt]=int(result['scores'].numel())
                        if result['scores'].numel():
                            evidence=(result['masks_logits'].float().squeeze(1)*result['scores'].float()[:,None,None]).amax(dim=0).cpu().numpy()
                            scores[name]=np.maximum(scores[name],evidence)
                metal,nonmetal=scores['metal'],scores['nonmetal']
                labels=assign_material_pixels(metal,nonmetal,settings['pixel_threshold'],settings['conflict_margin'])
            Image.fromarray(labels).save(output/(stem+'.png'))
            np.savez_compressed(output/(stem+'.npz'),metal=np.where(labels==2,metal,0).astype(np.float16),
                                nonmetal=np.where(labels==1,nonmetal,0).astype(np.float16),
                                raw_metal=metal.astype(np.float16),raw_nonmetal=nonmetal.astype(np.float16))
            save_previews(output,stem,image,labels,details)
            record=dict(image=str(image_path.relative_to(scene)),image_sha256=image_hash,width=image.width,height=image.height,
                        labels=stem+'.png',scores=stem+'.npz',labels_sha256=file_sha256(output/(stem+'.png')),
                        scores_sha256=file_sha256(output/(stem+'.npz')),
                        pixel_counts={name:int((labels==label).sum()) for name,label in config['labels'].items()},
                        seconds=time.perf_counter()-start,**details)
            header['frames'][stem]=record
            atomic_json(manifest_path,header)
            print('[{}/{}] {} {}'.format(index+1,len(images),stem,record['pixel_counts']),flush=True)
    header['completed']=True
    atomic_json(manifest_path,header)
    print('Material masks:',manifest_path,flush=True)


if __name__=='__main__':
    main()
