#!/usr/bin/env python3
"""Fine SAM3 material hypotheses from RGB TRAIN images; separate from legacy 3-ID masks."""
import argparse,base64,hashlib,json,os,time
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_json(path,obj):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)

def compose(instances,shape,config):
    labels=np.zeros(shape,np.uint8);confidence=np.zeros(shape,np.float32)
    priority=np.full(shape,-1,np.int16);parent=np.zeros(shape,bool);conflict=np.zeros(shape,bool)
    for level in sorted({v['priority'] for v in instances}):
        maps={};caps={};parents={}
        for item in instances:
            if item['priority']!=level:continue
            c=item['label_id'];x0,y0,x1,y1=item['window']
            raw=item['probabilities'];e=np.where(raw>=config['mask_threshold'],raw*item['score'],0)
            dst=maps.setdefault(c,np.zeros(shape,np.float32))[y0:y1,x0:x1]
            cap=caps.setdefault(c,np.zeros(shape,np.float32))[y0:y1,x0:x1]
            p=parents.setdefault(c,np.zeros(shape,bool))[y0:y1,x0:x1]
            better=e>dst
            cap[better]=item.get('confidence_cap',1.);p[better]=item.get('parent_fallback',False)
            np.maximum(dst,e,out=dst)
        ids=np.array(sorted(maps));stack=np.stack([maps[k] for k in ids]);which=stack.argmax(0)
        best=np.take_along_axis(stack,which[None],axis=0)[0]
        runner=np.partition(stack,-2,axis=0)[-2] if len(ids)>1 else np.zeros(shape,np.float32)
        present=best>=config['evidence_threshold']
        for i,c in enumerate(ids):
            selected=present&(which==i)
            labels[selected]=c;confidence[selected]=np.minimum(best[selected],caps[c][selected]);parent[selected]=parents[c][selected]
        priority[present]=level
        conflict[present]=(runner[present]>0)&((best-runner)[present]<config['conflict_margin'])
    uncertain=(labels==0)|(confidence<.30)|parent|conflict|(labels==config['labels']['metal_finish_uncertain']['id'])
    return labels,confidence,priority,parent,conflict,uncertain


def infer(processor,image,config):
    groups=config['groups'];instances=[];detections=[];anchors=[]
    W,H=image.size
    def window_inference(window,kind):
        x0,y0,x1,y1=window;state=processor.set_image(image.crop(window))
        for group in groups:
            if kind not in group['views']:continue
            processor.set_confidence_threshold(group.get('instance_threshold',config['instance_threshold']))
            for prompt in group['prompts']:
                processor.reset_all_prompts(state)
                result=processor.set_text_prompt(prompt=prompt,state=state)
                masks=result['masks_logits'].float().squeeze(1).cpu().numpy()
                scores=result['scores'].float().cpu().numpy();boxes=result['boxes'].float().cpu().numpy()
                accepted=0
                for mask,score,box in zip(masks,scores,boxes):
                    if group['name']=='sky':
                        rgb=np.asarray(image.crop(window),dtype=np.float32)/255.
                        mask=mask.copy();mask[rgb.mean(-1)<.25]=0
                    area=int((mask>=config['mask_threshold']).sum())
                    if area<6 or area>W*H*group.get('max_fraction',1.) or not np.isfinite(mask).all():continue
                    box=(box+np.array([x0,y0,x0,y0])).tolist()
                    item=dict(group=group['name'],label=group['label'],label_id=config['labels'][group['label']]['id'],
                        priority=group['priority'],prompt=prompt,score=float(score),window=list(window),box=box,
                        probabilities=mask,mask_pixels=area,view=kind,
                        confidence_cap=group.get('confidence_cap',1.),parent_fallback=group.get('parent_fallback',False))
                    instances.append(item);accepted+=1
                    if kind=='full' and ((group.get('anchor')=='cab' and prompt=='truck cab') or group.get('anchor')=='wheel'):
                        anchors.append(item)
                detections.append(dict(prompt=prompt,group=group['name'],view=kind,window=list(window),detected=len(scores),accepted=accepted))
        del state
    window_inference((0,0,W,H),'full')
    crops=[]
    candidates=sorted(anchors,key=lambda x:(x['group']!='painted_panels',-x['score']))
    for item in candidates:
        a,b,c,d=item['box'];bw,bh=c-a,d-b
        if bw<20 or bh<20:continue
        pad=.14 if item['group']=='painted_panels' else .45
        side=max(bw,bh)*(1+2*pad);side=max(100,min(side,min(W,H)))
        cx,cy=(a+c)/2,(b+d)/2
        left=int(np.clip(cx-side/2,0,W-side));top=int(np.clip(cy-side/2,0,H-side))
        crop=(left,top,int(left+side),int(top+side))
        if side>min(W,H)*.93:continue
        if any(sum(abs(x-y) for x,y in zip(crop,old))<60 for old in crops):continue
        crops.append(crop)
        if len(crops)>=config['max_crops']:break
    for crop in crops:window_inference(crop,'crop')
    # Vehicle part prompts may respond to furnace rectangles or floor glare.
    # Keep those hypotheses only inside a text-detected vehicle silhouette.
    from scipy.ndimage import binary_dilation
    truck=np.zeros((H,W),bool)
    for item in instances:
        if item['view']=='full' and item['group']=='truck_parent':
            truck |= item['probabilities']>=config['mask_threshold']
    if truck.any():
        truck=binary_dilation(truck,iterations=4)
        vehicle_parts={'bare_metal','tires','rims','plastic_trim','lamps','windows'}
        kept=[]
        for item in instances:
            if item['group'] in vehicle_parts:
                x0,y0,x1,y1=item['window']
                item['probabilities']=item['probabilities']*truck[y0:y1,x0:x1]
                item['mask_pixels']=int((item['probabilities']>=config['mask_threshold']).sum())
                if item['mask_pixels']<6:continue
            kept.append(item)
        instances=kept
    return instances,dict(detections=detections,crops=crops)


def preview(path,stem,image,labels,confidence,uncertain,config):
    W,H=image.size;palette=np.zeros((256,3),np.uint8)
    for v in config['labels'].values():palette[v['id']]=v['color']
    color=palette[labels];overlay=(.62*np.array(image)+.38*color).astype(np.uint8)
    Image.fromarray(color).save(path/(stem+'.labels_rgb.png'))
    Image.fromarray(overlay).save(path/(stem+'.overlay.jpg'),quality=94)
    canvas=Image.new('RGB',(W*3,H+210),'white');draw=ImageDraw.Draw(canvas)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
    for i,(im,title) in enumerate(((image,'RGB'),(Image.fromarray(color),'Fine material labels'),(Image.fromarray(overlay),'Overlay'))):
        canvas.paste(im,(i*W,32));draw.text((i*W+8,8),stem+' | '+title,fill='black',font=font)
    present=[(k,v) for k,v in config['labels'].items() if (labels==v['id']).any()]
    for i,(name,v) in enumerate(present):
        col=i%4;row=i//4;x=col*(W*3//4)+8;y=H+44+row*25
        draw.rectangle((x,y,x+18,y+17),fill=tuple(v['color']))
        pct=100*float((labels==v['id']).mean())
        draw.text((x+25,y),f"{v['id']:02d} {name} {pct:.1f}%",fill='black',font=font)
    draw.text((8,H+184),f'Unknown={100*(labels==0).mean():.2f}% | uncertain={100*uncertain.mean():.2f}% | visual hypotheses; inspect surface finish',fill='black',font=font)
    canvas.save(path/(stem+'.qa.jpg'),quality=94)


def save_frame(output,image_path,scene,config,processor):
    stem=image_path.stem;image=Image.open(image_path).convert('RGB');start=time.monotonic()
    instances,details=infer(processor,image,config)
    labels,confidence,priority,parent,conflict,uncertain=compose(instances,(image.height,image.width),config)
    Image.fromarray(labels).save(output/(stem+'.png'))
    np.save(output/(stem+'.confidence.npy'),confidence.astype(np.float16))
    coarse=np.zeros_like(labels)
    for item in config['labels'].values():coarse[labels==item['id']]=item['coarse']
    sky=config['labels']['sky']['id']
    geometry_valid=(labels!=0)&(labels!=sky)
    prior_valid=geometry_valid&(~uncertain)
    np.savez_compressed(output/(stem+'.npz'),confidence=confidence.astype(np.float16),
        uncertain=uncertain,parent_fallback=parent,conflict=conflict,priority=priority,
        geometry_valid=geometry_valid,material_prior_valid=prior_valid,coarse_labels=coarse,
        metal=np.where(coarse==2,confidence,0).astype(np.float16),nonmetal=np.where(coarse==1,confidence,0).astype(np.float16))
    # Retain individual probabilistic masks for review/recomposition without rerunning SAM3.
    stored={f'm_{i:04d}':np.round(np.clip(item['probabilities'],0,1)*255).astype(np.uint8) for i,item in enumerate(instances)}
    np.savez_compressed(output/(stem+'.instances.npz'),**stored)
    records=[dict({k:v for k,v in item.items() if k!='probabilities'},mask_key=f'm_{i:04d}') for i,item in enumerate(instances)]
    write_json(output/(stem+'.instances.json'),dict(instances=records,**details))
    preview(output,stem,image,labels,confidence,uncertain,config)
    return dict(image=str(image_path.relative_to(scene)),image_sha256=sha(image_path),width=image.width,height=image.height,
        labels=stem+'.png',labels_sha256=sha(output/(stem+'.png')),scores=stem+'.npz',scores_sha256=sha(output/(stem+'.npz')),
        confidence=stem+'.confidence.npy',confidence_sha256=sha(output/(stem+'.confidence.npy')),
        instances=stem+'.instances.json',instance_masks=stem+'.instances.npz',
        pixel_counts={k:int((labels==v['id']).sum()) for k,v in config['labels'].items()},
        unknown_fraction=float((labels==0).mean()),uncertain_fraction=float(uncertain.mean()),
        material_prior_valid_fraction=float(prior_valid.mean()),instances_count=len(instances),seconds=time.monotonic()-start)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scene',required=True);p.add_argument('--config',required=True)
    p.add_argument('--output');p.add_argument('--names',nargs='+');p.add_argument('--worker',type=int,default=0);p.add_argument('--workers',type=int,default=1)
    p.add_argument('--resume',action='store_true');p.add_argument('--merge-only',action='store_true')
    a=p.parse_args();scene=Path(a.scene).resolve();cfgpath=Path(a.config).resolve();config=json.loads(cfgpath.read_text())
    paths=sorted(p for p in (scene/'rgb/train').iterdir() if p.suffix.lower() in ('.jpg','.jpeg','.png'))
    if a.names:paths=[p for p in paths if p.stem in a.names]
    if not paths or len({p.stem for p in paths})!=len(paths):raise ValueError('Invalid RGB training images')
    if a.names and len(paths)!=len(set(a.names)):raise ValueError('Some requested names are not train images')
    output=Path(a.output).resolve() if a.output else scene/config['output_directory'];output.mkdir(parents=True,exist_ok=True)
    header=dict(schema_version='sam3_fine_material_masks_v1',scene=str(scene),split='train',
        checkpoint_sha256=sha(config['checkpoint']),generator_sha256=sha(__file__),config_path=str(cfgpath),config_sha256=sha(cfgpath),labels=config['labels'],
        prompts=config['groups'],expected_frames=[p.stem for p in paths],completed=False,frames={})
    header_path=output/f'manifest.worker_{a.worker}.json'
    if a.merge_only:
        for part in output.glob('manifest.worker_*.json'):
            record=json.loads(part.read_text())
            for k in ('config_sha256','generator_sha256','scene','split'):
                if record[k]!=header[k]:raise ValueError('Incompatible worker result: '+str(part))
            for stem,frame in record['frames'].items():
                if stem in header['frames']:raise ValueError('Duplicate worker image '+stem)
                header['frames'][stem]=frame
        if set(header['frames'])!={p.stem for p in paths}:raise ValueError('Incomplete worker results')
        for frame in header['frames'].values():
            if sha(output/frame['labels'])!=frame['labels_sha256'] or sha(output/frame['scores'])!=frame['scores_sha256']:raise ValueError('Corrupt result')
        header['completed']=True;write_json(output/'manifest.json',header)
        print('MERGED '+str(output/'manifest.json'),flush=True);return
    if not 0<=a.worker<a.workers:raise ValueError('Invalid worker shard')
    if header_path.exists():
        if not a.resume:raise FileExistsError('Use --resume for verified matching outputs')
        old=json.loads(header_path.read_text())
        for k in ('config_sha256','generator_sha256','scene','split','expected_frames'):
            if old[k]!=header[k]:raise ValueError('Incompatible resume settings '+k)
        header=old
    write_json(header_path,header)
    os.environ['HF_HUB_OFFLINE']='1'
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    torch.set_num_threads(4)
    model=build_sam3_image_model(checkpoint_path=config['checkpoint'],load_from_HF=False,compile=False)
    processor=Sam3Processor(model,confidence_threshold=config['instance_threshold'])
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print('SAM3 worker',a.worker,'device',torch.cuda.get_device_name(0),'precision',dtype,flush=True)
    with torch.inference_mode(),torch.autocast('cuda',dtype=dtype):
      for path in paths[a.worker::a.workers]:
        if path.stem in header['frames']:
            old=header['frames'][path.stem]
            if sha(path)!=old['image_sha256'] or sha(output/old['labels'])!=old['labels_sha256'] or sha(output/old['scores'])!=old['scores_sha256']:raise ValueError('Stale frame cache')
            continue
        record=save_frame(output,path,scene,config,processor)
        header['frames'][path.stem]=record;write_json(header_path,header)
        print(json.dumps(dict(worker=a.worker,frame=path.stem,**{k:record[k] for k in ('seconds','unknown_fraction','uncertain_fraction','material_prior_valid_fraction','instances_count','pixel_counts')})),flush=True)
    header['completed']=True;write_json(header_path,header)
    if a.workers==1:
        header['completed']=len(header['frames'])==len(paths)
        write_json(output/'manifest.json',header)
    print('FINISHED worker',a.worker,flush=True)

if __name__=='__main__':main()
