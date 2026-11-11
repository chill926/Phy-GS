#!/usr/bin/env python3
"""Export fine SAM3 instances to the existing 0/1/2 material-mask contract."""
import argparse,copy,csv,hashlib,json,sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.segment_sam3_fine_materials import compose

COLORS=np.array([[90,90,90],[60,200,90],[30,110,255]],np.uint8)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def js(path,data):
 path=Path(path);temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');temp.replace(path)
def qa(path,stem,rgb,labels,uncertain):
 W,H=rgb.size;colors=COLORS[labels];overlay=(.65*np.array(rgb)+.35*colors).astype(np.uint8)
 Image.fromarray(colors).save(path/(stem+'.labels_rgb.png'))
 Image.fromarray(overlay).save(path/(stem+'.overlay.jpg'),quality=95)
 canvas=Image.new('RGB',(W*3,H+102),'white');d=ImageDraw.Draw(canvas)
 font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
 for i,(im,title) in enumerate(((rgb,'RGB'),(Image.fromarray(colors),'0/1/2 labels'),(Image.fromarray(overlay),'Overlay'))):
  canvas.paste(im,(i*W,30));d.text((i*W+10,5),stem+' | '+title,fill='black',font=font)
 d.text((10,H+40),'blue=exposed metal (2) | green=nonmetal / paint / oxide (1) | gray=unknown (0)',fill='black',font=font)
 d.text((10,H+70),f'unknown={100*(labels==0).mean():.2f}% | uncertain surface/boundary={100*uncertain.mean():.2f}% | inspect wheel rims and small hardware',fill='black',font=font)
 canvas.save(path/(stem+'.qa.jpg'),quality=95)


def guarded_parts(fine_dir,record,config,rgb):
    from scipy.ndimage import label as components
    details=json.loads((fine_dir/record['instances']).read_text())
    instances=[];H,W=rgb.shape[:2]
    with np.load(fine_dir/record['instance_masks'],allow_pickle=False) as masks:
        for raw in details['instances']:
            item=dict(raw);item['probabilities']=masks[item['mask_key']].astype(np.float32)/255.
            instances.append(item)
    tires=np.zeros((H,W),bool)
    for item in instances:
        if item['group']=='tires':
            x0,y0,x1,y1=item['window'];tires[y0:y1,x0:x1] |= item['probabilities']>=config['mask_threshold']
    tire_instances=[]
    for item in instances:
        if item['group']=='tires':
            x0,y0,x1,y1=item['window'];full=np.zeros((H,W),bool)
            full[y0:y1,x0:x1]=item['probabilities']>=config['mask_threshold']
            tire_instances.append(full)
    cc,n=components(tires);areas=np.bincount(cc.ravel())
    stats={'rim_masks_guarded':0,'rim_pixels_rejected':0,'grille_pixels_rejected':0}
    gray=(.299*rgb[...,0]+.587*rgb[...,1]+.114*rgb[...,2]).astype(np.uint8)
    saturation=(rgb.max(-1).astype(float)-rgb.min(-1))/np.maximum(rgb.max(-1),1)
    for item in instances:
        x0,y0,x1,y1=item['window'];prob=item['probabilities'];mask=prob>=config['mask_threshold']
        before=int(mask.sum())
        if item['group']=='rims' and n:
            full_rim=np.zeros((H,W),bool);full_rim[y0:y1,x0:x1]=mask
            whole_wheel=False
            for tire in tire_instances:
                intersection=int((full_rim&tire).sum())
                union=int(full_rim.sum()+tire.sum()-intersection)
                if intersection/max(union,1)>=.65:
                    whole_wheel=True;break
            if whole_wheel:
                prob[:]=0;stats['rim_masks_guarded']+=1;stats['rim_pixels_rejected']+=before
                continue
            sub=cc[y0:y1,x0:x1];intersections=np.bincount(sub[mask],minlength=n+1)
            bad=[i for i in range(1,n+1) if areas[i]>=20 and intersections[i]/areas[i]>.65
                 and intersections[i]/max(before,1)>.45]
            if bad:
                # Reject a whole-wheel response; retain the existing tire or
                # body evidence, rather than inventing an elliptical rim.
                rejected=np.isin(sub,bad)&mask
                prob[rejected]=0;stats['rim_masks_guarded']+=1;stats['rim_pixels_rejected']+=int(rejected.sum())
        if item['prompt']=='chrome truck grille' and before>=30:
            local=gray[y0:y1,x0:x1];values=local[mask]
            hist=np.bincount(values,minlength=256).astype(float);weight=hist.cumsum();moment=(hist*np.arange(256)).cumsum()
            denom=weight*(weight[-1]-weight)
            variance=np.divide((moment[-1]*weight-moment*weight[-1])**2,denom,out=np.zeros(256),where=denom>0)
            threshold=max(int(variance.argmax()),int(np.median(values)))
            keep=(local>=threshold)&(saturation[y0:y1,x0:x1]<.35)
            rejected=mask&~keep;prob[rejected]=0;stats['grille_pixels_rejected']+=int(rejected.sum())
    labels,confidence,priority,parent,conflict,uncertain=compose(instances,(H,W),config)
    return labels,confidence,parent,conflict,uncertain,stats


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--scene',required=True);p.add_argument('--fine-config',required=True)
 p.add_argument('--base-config',default=str(ROOT/'data/materials/sam3_materials.json'));p.add_argument('--config-output',default=str(ROOT/'data/materials/sam3_truck_materials.json'))
 p.add_argument('--resume',action='store_true');a=p.parse_args()
 scene=Path(a.scene).resolve();finepath=Path(a.fine_config).resolve();fine=json.loads(finepath.read_text())
 fine_dir=scene/fine['output_directory'];source_manifest=fine_dir/'manifest.json';source=json.loads(source_manifest.read_text())
 if source.get('completed') is not True or source.get('split')!='train' or source['config_sha256']!=sha(finepath):raise ValueError('Require a complete matching fine segmentation')
 train={v.stem for v in (scene/'rgb/train').iterdir() if v.suffix.lower() in ('.jpg','.png','.jpeg')}
 test={v.stem for v in (scene/'rgb/test').iterdir() if v.suffix.lower() in ('.jpg','.png','.jpeg')}
 if set(source['frames'])!=train or train&test:raise ValueError('Mask split/name mismatch')
 config=copy.deepcopy(json.loads(Path(a.base_config).read_text()));config.pop('mask_source_config',None)
 groups=[]
 for g in fine['groups']:
  coarse=fine['labels'][g['label']]['coarse']
  if coarse==0:continue
  group={k:v for k,v in g.items() if k not in ('label','views','anchor','confidence_cap','parent_fallback','max_fraction')}
  group.update({'class':'metal' if coarse==2 else 'nonmetal','views':['anchor_base' if v=='crop' else v for v in g['views']],
                'instance_threshold':g.get('instance_threshold',fine['instance_threshold'])})
  if 'max_fraction' in g:group['max_mask_fraction']=g['max_fraction']
  groups.append(group)
 sam=config['sam3'];sam.update(mask_directory='material_masks/sam3_v2',checkpoint=fine['checkpoint'],pipeline_version=2,
       mask_probability_threshold=fine['mask_threshold'],pixel_threshold=fine['evidence_threshold'],
       instance_threshold=fine['instance_threshold'],conflict_margin=fine['conflict_margin'],groups=groups,
       scene_groups={'Truck':groups},scene_prompts={})
 sam['prompts']={key:sorted({p for g in groups if g['class']==key for p in g['prompts']}) for key in ('metal','nonmetal')}
 config['fine_segmentation']=dict(postprocessing=dict(rim_tire_instance_iou_max=.65,rim_parent_overlap_max=.65,rim_covered_fraction_min=.45,chrome_grille='SAM3 mask intersect achromatic bright bars by local Otsu/median; darker grille cavities retain parent nonmetal'),generator='tools/segment_sam3_fine_materials.py',exporter='tools/export_sam3_fine_materials.py',
    config_path=str(finepath),config_sha256=sha(finepath),source_manifest=str(source_manifest),source_manifest_sha256=sha(source_manifest),
    material_mapping=fine['coarse_mapping_policy'],surface_mapping_user_confirmed=True,
    score_meaning='SAM3 evidence and conservative caps; not calibrated physical uncertainty')
 configpath=Path(a.config_output).resolve();configpath.parent.mkdir(parents=True,exist_ok=True)
 if configpath.exists() and json.loads(configpath.read_text())!=config and not a.resume:raise FileExistsError('Config output already exists with different settings')
 js(configpath,config)
 output=scene/sam['mask_directory'];output.mkdir(parents=True,exist_ok=True)
 manifestpath=output/'manifest.json'
 manifest=dict(schema_version=1,generator_version=3,generator='SAM3 fine instances plus surface-material mapping',
    generator_sha256=sha(__file__),fine_generator_sha256=source['generator_sha256'],
    scene=str(scene),split='train',labels={'unknown':0,'nonmetal':1,'metal':2},
    config_path=str(configpath),config_sha256=sha(configpath),checkpoint_sha256=source['checkpoint_sha256'],
    fine_source_manifest=str(source_manifest),fine_source_manifest_sha256=sha(source_manifest),
    prompts=groups,completed=False,frames={})
 if manifestpath.exists():
  if not a.resume:raise FileExistsError('Output already has a manifest; use --resume')
  old=json.loads(manifestpath.read_text())
  if any(old.get(k)!=manifest[k] for k in ('config_sha256','fine_source_manifest_sha256','generator_sha256')):raise ValueError('Incompatible resume manifest')
  manifest=old;manifest['completed']=False
 js(manifestpath,manifest)
 summary=[]
 for stem,record in sorted(source['frames'].items()):
  imagepath=scene/record['image']
  if sha(imagepath)!=record['image_sha256'] or sha(fine_dir/record['labels'])!=record['labels_sha256']:raise ValueError('Stale source image or fine labels '+stem)
  if stem in manifest['frames']:
   old=manifest['frames'][stem]
   if sha(output/old['labels'])!=old['labels_sha256'] or sha(output/old['scores'])!=old['scores_sha256']:raise ValueError('Corrupt resumed frame '+stem)
   continue
  rgb=Image.open(imagepath).convert('RGB')
  labels_fine,confidence,parent,conflict,uncertain,guard_stats=guarded_parts(fine_dir,record,fine,np.array(rgb))
  Image.fromarray(labels_fine).save(output/(stem+'.parts.png'))
  lut=np.zeros(256,np.uint8)
  for v in fine['labels'].values():lut[v['id']]=v['coarse']
  labels=lut[labels_fine]
  confidence=np.where(labels>0,confidence,0)
  metal=np.where(labels==2,confidence,0).astype(np.float16);nonmetal=np.where(labels==1,confidence,0).astype(np.float16)
  Image.fromarray(labels).save(output/(stem+'.png'));np.save(output/(stem+'.confidence.npy'),confidence.astype(np.float16))
  np.savez_compressed(output/(stem+'.npz'),metal=metal,nonmetal=nonmetal,raw_metal=metal,raw_nonmetal=nonmetal,
      confidence=confidence.astype(np.float16),uncertain=uncertain,parent_fallback=parent,conflict=conflict)
  rgb=Image.open(imagepath).convert('RGB');qa(output,stem,rgb,labels,uncertain)
  # Assess unknown coverage within text-detected vehicle silhouettes separately
  # from dark industrial void/background; no test views are used.
  vehicle=np.zeros(labels.shape,bool)
  details=json.loads((fine_dir/record['instances']).read_text())
  with np.load(fine_dir/record['instance_masks'],allow_pickle=False) as masks:
   for item in details['instances']:
    if item['group']=='truck_parent' and item['view']=='full':vehicle |= masks[item['mask_key']]>=128
  counts={k:int((labels==v).sum()) for k,v in manifest['labels'].items()}
  row=dict(image=record['image'],image_sha256=record['image_sha256'],width=record['width'],height=record['height'],
      labels=stem+'.png',labels_sha256=sha(output/(stem+'.png')),scores=stem+'.npz',scores_sha256=sha(output/(stem+'.npz')),
      confidence=stem+'.confidence.npy',confidence_sha256=sha(output/(stem+'.confidence.npy')),pixel_counts=counts,
      unknown_fraction=float((labels==0).mean()),uncertain_fraction=float(uncertain.mean()),
      vehicle_pixels=int(vehicle.sum()),vehicle_unknown_fraction=float((labels[vehicle]==0).mean()) if vehicle.any() else None,
      guard_statistics=guard_stats,parts=stem+'.parts.png',
      fine_labels=str((fine_dir/record['labels']).relative_to(scene)),fine_instances=str((fine_dir/record['instances']).relative_to(scene)))
  manifest['frames'][stem]=row;js(manifestpath,manifest)
  print(stem,counts,'vehicle unknown',row['vehicle_unknown_fraction'],flush=True)
 manifest['completed']=len(manifest['frames'])==len(train);js(manifestpath,manifest)
 total=sum(v['width']*v['height'] for v in manifest['frames'].values());counts={k:sum(v['pixel_counts'][k] for v in manifest['frames'].values()) for k in manifest['labels']}
 vehicle_total=sum(v['vehicle_pixels'] for v in manifest['frames'].values());vehicle_unknown=sum(v['vehicle_pixels']*v['vehicle_unknown_fraction'] for v in manifest['frames'].values() if v['vehicle_unknown_fraction'] is not None)
 report=dict(scene='Truck',frames=len(manifest['frames']),expected_train_frames=len(train),test_frames_used=0,
     labels=manifest['labels'],pixel_counts=counts,pixel_fractions={k:v/total for k,v in counts.items()},
     vehicle_unknown_fraction=vehicle_unknown/max(vehicle_total,1),config_path=str(configpath),manifest=str(manifestpath),
     quality_status='automatic segmentation complete; selected frames awaiting user review',surface_mapping=config['fine_segmentation']['material_mapping'])
 js(output/'mask_quality_report.json',report)
 fields=['frame','unknown_fraction','uncertain_fraction','vehicle_unknown_fraction','vehicle_pixels','metal','nonmetal','unknown']
 with (output/'frame_statistics.csv').open('w',newline='') as file:
  w=csv.DictWriter(file,fieldnames=fields);w.writeheader()
  for stem,v in sorted(manifest['frames'].items()):w.writerow(dict(frame=stem,**{k:v[k] for k in fields[1:5]},**v['pixel_counts']))
 print('EXPORTED',json.dumps(report,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
