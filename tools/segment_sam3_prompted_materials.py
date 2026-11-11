#!/usr/bin/env python3
"""SAM3 text+interactive material candidates: all final contours come from SAM."""
import argparse,hashlib,json,os,time
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch
from scipy.ndimage import distance_transform_edt
ROOT=Path(__file__).resolve().parents[1]
BANK=[
 ('painted metal surface',1,.4),('rusty metal',1,.4),('rusted steel surface',1,.4),('industrial equipment',1,.35),('steel structure',1,.35),('metal roof',1,.35),('factory ceiling',1,.35),('metal platform',1,.35),('wooden board',1,.4),('dirt ground',1,.55),('muddy ground',1,.55),('concrete pillar',1,.4),
 ('ground',1,.65),('concrete floor',1,.65),('concrete wall',1,.55),('brick wall',1,.55),('pile of sand',1,.7),('pile of gravel',1,.65),
 ('truck',1,.5),('red truck body',1,.55),('truck cab',1,.6),('cylindrical tank',1,.4),('large metal container',1,.4),
 ('vertical pipe',1,.4),('metal railing',1,.4),('industrial wall',1,.4),('corrugated metal roof',1,.4),
 ('truck windshield',1,.8),('truck side window',1,.8),('rubber tire',1,.75),('truck tire',1,.75),('truck headlight',1,.7),('plastic cover',1,.5),
 ('truck fuel tank',2,.5),('metal truck step',2,.5),('metal bolt',2,.5),('metal screw',2,.5)]
REVIEWED={
 'FLIR3110':[dict(name='visible oxidized vessel',box=[90,30,400,479],positive=[[280,315],[200,270],[300,110]],negative=[[450,410]])],
 'FLIR3100':[dict(name='coated vessel behind cab',box=[170,0,390,230],positive=[[280,100],[270,40],[240,160]],negative=[[410,210]])]
}
COLORS=np.array([[90,90,90],[60,200,90],[30,110,255]],np.uint8)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p,v):
 p=Path(p);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');q.replace(p)
def fps(mask,n):
 yy,xx=np.where(mask)
 if len(xx)==0:return []
 dist=distance_transform_edt(mask);inside=dist[yy,xx]>=min(2,float(dist.max()))
 xx,yy=xx[inside],yy[inside]
 if not len(xx):return []
 pts=np.stack([xx,yy],1);idx=int(np.argmax(dist[yy,xx]));out=[pts[idx]]
 for _ in range(n-1):
  d=np.min(np.stack([((pts-a)**2).sum(1) for a in out]),axis=0)
  idx=int(np.argmax(d))
  if d[idx]<36:break
  out.append(pts[idx])
 return [p.tolist() for p in out]
def box_iou(a,b):
 inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
 return inter/max((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter,1)
def compose(proposals,shape):
 labels=np.zeros(shape,np.uint8);conf=np.zeros(shape,np.float32);priority=np.full(shape,-1,np.int16)
 for item in sorted(proposals,key=lambda v:(v['priority'],v['confidence'])):
  selected=item['mask']&((priority<item['priority'])|((priority==item['priority'])&(conf<item['confidence'])))
  labels[selected]=item['label'];conf[selected]=item['confidence'];priority[selected]=item['priority']
 return labels,conf

def process(model,processor,scene,name,out):
 from sam3.model.sam3_image_processor import Sam3Processor
 imagepath=scene/'rgb/train'/(name+'.jpg');rgb=Image.open(imagepath).convert('RGB');W,H=rgb.size;start=time.monotonic()
 proposals=[];detections=[];anchors=[];point_records=[]
 state=processor.set_image(rgb)
 for prompt,lab,cap in BANK:
  processor.set_confidence_threshold(.23 if lab==1 else .4);processor.reset_all_prompts(state);r=processor.set_text_prompt(prompt=prompt,state=state)
  ps=r['masks_logits'].float().squeeze(1).cpu().numpy();scores=r['scores'].float().cpu().numpy();boxes=r['boxes'].float().cpu().numpy()
  for prob,score,box in zip(ps,scores,boxes):
   mask=prob>=.5;area=int(mask.sum())
   if area<10 or area>W*H*.88:continue
   if lab==2 and area>W*H*(.025 if prompt in ('metal bolt','metal screw') else .15):continue
   proposals.append(dict(mask=mask,label=lab,confidence=min(float(score),cap),priority=20 if lab==2 else 10,method='SAM3_text',prompt=prompt,score=float(score)))
   box=np.array(box);box[[0,2]]=np.clip(box[[0,2]],0,W-1);box[[1,3]]=np.clip(box[[1,3]],0,H-1)
   detections.append(dict(prompt=prompt,box=box.tolist(),score=float(score),area=area))
   if prompt in ('cylindrical tank','large metal container','vertical pipe','industrial wall') and area>W*H*.015:
    anchors.append(dict(box=box.copy(),mask=mask.copy(),prompt=prompt,score=float(score)))
 base_labels,base_conf=compose(proposals,(H,W))
 # Previous SAM instances are used only as visual prompt coordinates, never as
 # final pixel masks. No color/brightness threshold defines any final contour.
 previous=scene/'material_masks/sam3_fine_v1';old_json=json.loads((previous/(name+'.instances.json')).read_text())
 source=[];truck=np.zeros((H,W),bool)
 with np.load(previous/(name+'.instances.npz'),allow_pickle=False) as z:
  for item in old_json['instances']:
   x0,y0,x1,y1=item['window'];mask=np.zeros((H,W),bool);mask[y0:y1,x0:x1]=z[item['mask_key']]>=128
   raw=dict(item,mask=mask);source.append(raw)
   if item['prompt']=='truck' and item['view']=='full':truck|=mask
 # Merge vertically aligned SAM-detected coated panels with tank proposals.
 extras=[i for i in source if i['view']=='full' and i['prompt'] in ('wooden board','painted metal surface','rusty metal','rusted steel surface') and i['mask_pixels']>250]
 for a in anchors:
  if a['prompt'] not in ('cylindrical tank','large metal container'):continue
  for _ in range(3):
   for item in extras:
    if (item['mask']&truck).sum()/max(item['mask'].sum(),1)>.35:continue
    b=a['box'];c=np.array(item['box']);bw=b[2]-b[0];cw=c[2]-c[0]
    overlap=max(0,min(b[2],c[2])-max(b[0],c[0]));gap=max(0,c[1]-b[3],b[1]-c[3])
    if overlap>.5*min(bw,cw) and abs((b[0]+b[2]-c[0]-c[2])/2)<.35*max(bw,cw) and gap<max(80,.3*(b[3]-b[1])):
     a['box']=np.array([min(b[0],c[0]),min(b[1],c[1]),max(b[2],c[2]),max(b[3],c[3])]);a['mask']|=item['mask']
 chosen=[]
 for a in sorted(anchors,key=lambda v:((v['mask']&(base_labels==0)).sum(),v['score']),reverse=True):
  b=a['box'];pad=.10*max(b[2]-b[0],b[3]-b[1]);box=np.array([max(0,b[0]-pad),max(0,b[1]-pad),min(W-1,b[2]+pad),min(H-1,b[3]+pad)])
  if any(box_iou(box,old)>.7 for old in chosen):continue
  if len(chosen)>=6:break
  positives=fps(a['mask'],3);negatives=fps(truck&~a['mask'],2)
  if not positives:continue
  points=np.asarray(positives+negatives,dtype=np.float32);pointlabels=np.array([1]*len(positives)+[0]*len(negatives))
  masks,ious,_=model.predict_inst(state,point_coords=points,point_labels=pointlabels,box=box,multimask_output=True)
  selected=None
  for k in np.argsort(ious)[::-1]:
   mask=masks[k].astype(bool);area=int(mask.sum())
   if float(ious[k])<.60 or area<100 or area>W*H*.65:continue
   if np.mean([mask[int(y),int(x)] for x,y in positives])<.66:continue
   if negatives and np.mean([mask[int(y),int(x)] for x,y in negatives])>.1:continue
   selected=(mask,float(ious[k]));break
  point_records.append(dict(type='automatic_box_points',name=a['prompt'],box=box.tolist(),positive=positives,negative=negatives,accepted=selected is not None,quality_score=selected[1] if selected else None))
  if selected:
   proposals.append(dict(mask=selected[0],label=1,confidence=min(.45,.6*selected[1]),priority=9,method='SAM3_point_box',prompt=a['prompt'],score=selected[1]));chosen.append(box)
 for anno in REVIEWED.get(name,[]):
  positives=anno['positive'];negatives=anno['negative'];points=np.asarray(positives+negatives,dtype=np.float32)
  masks,ious,_=model.predict_inst(state,point_coords=points,point_labels=np.array([1]*len(positives)+[0]*len(negatives)),box=np.asarray(anno['box']),multimask_output=True)
  k=int(np.argmax(ious));mask=masks[k].astype(bool)
  accepted=bool(float(ious[k])>=.6 and mask.sum()<W*H*.65)
  point_records.append(dict(type='reviewed_train_box_points',**anno,accepted=accepted,quality_score=float(ious[k])))
  if accepted:proposals.append(dict(mask=mask,label=1,confidence=.45,priority=9,method='SAM3_reviewed_points',prompt=anno['name'],score=float(ious[k])))
 # Re-segment small metal hypotheses using point prompts on a magnified image.
 # Old labels supply POSITIVE/NEGATIVE COORDINATES only; every output pixel is
 # generated afresh by SAM3's interactive segmentation head.
 old=np.array(Image.open(scene/'material_masks/sam3_v2'/(name+'.png')))
 tire_support=np.zeros((H,W),bool)
 for candidate in source:
  if candidate['group']=='tires':tire_support|=candidate['mask']
 local=[];attempted=[]
 for item in sorted(source,key=lambda v:v['score'],reverse=True):
  if not (item['prompt']=='chrome truck grille' or item['group']=='rims'):continue
  positive=(old==2)&item['mask']
  negative=(old==1)&(tire_support if item['group']=='rims' else item['mask'])
  if positive.sum()<20:continue
  b=np.array(item['box']);pad=.25*max(b[2]-b[0],b[3]-b[1]);box=np.array([max(0,b[0]-pad),max(0,b[1]-pad),min(W,b[2]+pad),min(H,b[3]+pad)]).astype(int)
  if any(box_iou(box,other)>.65 for other in attempted):continue
  if len(attempted)>=6:break
  x0,y0,x1,y1=box
  if x1-x0<10 or y1-y0<10:continue
  p=fps(positive[y0:y1,x0:x1],5);n=fps(negative[y0:y1,x0:x1],6)
  if not p or (item['group']=='rims' and not n):continue
  attempted.append(box)
  cs=processor.set_image(rgb.crop(tuple(box)));points=np.asarray(p+n,dtype=np.float32)
  masks,ious,_=model.predict_inst(cs,point_coords=points,point_labels=np.array([1]*len(p)+[0]*len(n)),multimask_output=True)
  selected=None
  for k in np.argsort(ious)[::-1]:
   mask=masks[k].astype(bool)
   minimum_quality=.45 if item['prompt']=='chrome truck grille' else .55
   if float(ious[k])<minimum_quality or mask.sum()<10:continue
   if np.mean([mask[int(y),int(x)] for x,y in p])<.6:continue
   if n and np.mean([mask[int(y),int(x)] for x,y in n])>.15:continue
   selected=(mask,float(ious[k]));break
  point_records.append(dict(type='metal_local_points',name=item['prompt'],crop=box.tolist(),positive=p,negative=n,accepted=selected is not None,quality_score=selected[1] if selected else None))
  if selected:
   full=np.zeros((H,W),bool);full[y0:y1,x0:x1]=selected[0]
   proposals.append(dict(mask=full,label=2,confidence=min(.6,.7*selected[1]),priority=40,method='SAM3_local_points',prompt=item['prompt'],score=selected[1]));local.append(box)
  del cs
 labels,confidence=compose(proposals,(H,W));np.savez_compressed(out/(name+'.proposals.npz'),**{f'm_{k:04d}':v['mask'] for k,v in enumerate(proposals)})
 Image.fromarray(labels).save(out/(name+'.png'));np.save(out/(name+'.confidence.npy'),confidence.astype(np.float16))
 np.savez_compressed(out/(name+'.npz'),metal=np.where(labels==2,confidence,0).astype(np.float16),nonmetal=np.where(labels==1,confidence,0).astype(np.float16),uncertain=(confidence<.45)|(labels==0))
 details=dict(proposals=[dict({k:v for k,v in x.items() if k!='mask'},mask_key=f'm_{i:04d}') for i,x in enumerate(proposals)],text_detections=detections,point_prompts=point_records)
 js(out/(name+'.prompts.json'),details)
 colors=COLORS[labels];oldcolors=COLORS[old];canvas=Image.new('RGB',(W*4,H+85),'white');d=ImageDraw.Draw(canvas);font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
 overlay=(.65*np.asarray(rgb)+.35*colors).astype(np.uint8)
 for k,(im,title) in enumerate(((rgb,'RGB'),(Image.fromarray(oldcolors),'previous masks'),(Image.fromarray(colors),'SAM3 text + point/box'),(Image.fromarray(overlay),'new overlay'))):
  canvas.paste(im,(W*k,30));d.text((W*k+7,6),name+' | '+title,font=font,fill='black')
 d.text((8,H+43),f'unknown old={100*(old==0).mean():.2f}% new={100*(labels==0).mean():.2f}% | all contours from SAM; no RGB-threshold mask cutting',font=font,fill='black')
 canvas.save(out/(name+'.qa.jpg'),quality=95)
 return dict(image='rgb/train/'+name+'.jpg',image_sha256=sha(imagepath),width=W,height=H,labels=name+'.png',labels_sha256=sha(out/(name+'.png')),scores=name+'.npz',scores_sha256=sha(out/(name+'.npz')),confidence=name+'.confidence.npy',confidence_sha256=sha(out/(name+'.confidence.npy')),pixel_counts={k:int((labels==v).sum()) for k,v in {'unknown':0,'nonmetal':1,'metal':2}.items()},unknown_fraction=float((labels==0).mean()),previous_unknown_fraction=float((old==0).mean()),prompts=name+'.prompts.json',seconds=time.monotonic()-start,point_masks_accepted=sum(x['accepted'] for x in point_records))

def main():
 global BANK,REVIEWED
 p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--output',required=True);p.add_argument('--names',nargs='+');p.add_argument('--prompt-config');p.add_argument('--worker',type=int,default=0);p.add_argument('--workers',type=int,default=1);a=p.parse_args()
 if a.prompt_config:
  selected=json.loads(Path(a.prompt_config).read_text());BANK=[(v['prompt'],int(v['label']),float(v['confidence_cap'])) for v in selected['text_prompts']];REVIEWED=selected.get('reviewed_train_prompts',{})
 scene=Path(a.scene).resolve();out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True)
 paths=sorted((scene/'rgb/train').glob('*.jpg'));names=[p.stem for p in paths]
 if a.names:names=[n for n in names if n in a.names]
 if not names or (a.names and set(names)!=set(a.names)):raise ValueError('Requested names must all be train views')
 if (out/f'manifest.worker_{a.worker}.json').exists():raise FileExistsError('Use a new result directory')
 os.environ['HF_HUB_OFFLINE']='1';from sam3 import build_sam3_image_model
 from sam3.model.sam3_image_processor import Sam3Processor
 torch.set_num_threads(4);model=build_sam3_image_model(checkpoint_path='/home/liupengyu/sam3/checkpoints/sam3.pt',load_from_HF=False,compile=False,enable_inst_interactivity=True);proc=Sam3Processor(model,confidence_threshold=.32)
 header=dict(prompt_config_path=str(Path(a.prompt_config).resolve()) if a.prompt_config else None,prompt_config_sha256=sha(a.prompt_config) if a.prompt_config else None,scene=str(scene),split='train',generator_sha256=sha(__file__),checkpoint_sha256=sha('/home/liupengyu/sam3/checkpoints/sam3.pt'),text_prompts=BANK,reviewed_train_prompts=REVIEWED,previous_manifest_sha256=sha(scene/'material_masks/sam3_v2/manifest.json'),fine_source_manifest_sha256=sha(scene/'material_masks/sam3_fine_v1/manifest.json'),all_contours_from_SAM=True,color_threshold_mask_cutting=False,frames={},completed=False)
 with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
  for n in names[a.worker::a.workers]:
   row=process(model,proc,scene,n,out);header['frames'][n]=row;js(out/f'manifest.worker_{a.worker}.json',header);print(n,row['seconds'],row['previous_unknown_fraction'],row['unknown_fraction'],row['point_masks_accepted'],flush=True)
 header['completed']=True;js(out/f'manifest.worker_{a.worker}.json',header)
 print('FINISHED',a.worker,flush=True)
if __name__=='__main__':main()
