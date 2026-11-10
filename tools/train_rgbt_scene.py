#!/usr/bin/env python3
"""CLI scene launcher: own DN adapter, unchanged main training, complete 20k state."""
import argparse
import ast
import csv
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from tools.full_training_state import (BOUNDARY,write_json,file_sha,transform_training,
    save_complete_checkpoint)


def resolve(value):
    path=Path(value).expanduser()
    return (ROOT/path).resolve() if not path.is_absolute() else path.resolve()


def scene_path(value):
    direct=resolve(value)
    return direct if direct.is_dir() else resolve('data/RGBT-Scenes/'+value)


def checked_run(command,env=None):
    print('[COMMAND] '+shlex.join([str(x) for x in command]),flush=True)
    subprocess.run([str(x) for x in command],cwd=ROOT,env=env,check=True)


def gpu_selection(index):
    lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,pci.bus_id,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'],text=True)
    rows=[]
    for row in csv.reader(lines.splitlines()):
        i,bus,memory,util=[s.strip() for s in row]
        rows.append(dict(index=int(i),bus=bus,memory_MiB=int(memory),utilization=int(util)))
    selected=next((r for r in rows if r['index']==index),None)
    if selected is None:raise ValueError('GPU index does not exist: '+str(index))
    ordered=sorted(rows,key=lambda r:tuple(int(p,16) for p in re.split('[:.]',r['bus'])))
    selected['cuda_ordinal']=next(i for i,r in enumerate(ordered) if r['index']==index)
    if selected['memory_MiB']>1024 or selected['utilization']>10:
        raise RuntimeError('GPU {} is busy; choose another --gpu'.format(index))
    return selected


def audit_scene(source,config,manifest,byte_order,threads):
    """Read-only dataset/mask checks. No fitting, parameter updates or CUDA."""
    import numpy as np
    import torch
    from PIL import Image
    from types import SimpleNamespace
    from utils.material_partition import resolve_material_inputs,_frame_data
    from tools.fit_flir_display_mapping import extract_flir
    torch.set_num_threads(threads)
    paired={}
    hashes={str(config):file_sha(config)}
    reference=config;seen=set()
    while True:
        reference=reference.resolve()
        if reference in seen:raise ValueError('Cycle in mask-source configurations')
        seen.add(reference);hashes[str(reference)]=file_sha(reference)
        origin=json.loads(reference.read_text()).get('mask_source_config')
        if not origin:break
        following=Path(origin).expanduser();reference=following if following.is_absolute() else reference.parent/following
    sparse=source/'sparse/0'
    if not sparse.is_dir():sparse=source/'colmap/sparse/0'
    for name in ('cameras.bin','images.bin','points3D.bin'):
        path=sparse/name
        if not path.is_file():raise FileNotFoundError('Missing COLMAP input: '+str(path))
        hashes[str(path)]=file_sha(path)
    if (sparse/'points3D.ply').is_file():hashes[str(sparse/'points3D.ply')]=file_sha(sparse/'points3D.ply')
    for split in ('train','test'):
        rgb={p.stem:p for p in (source/'rgb'/split).glob('*.jpg')}
        ir={p.stem:p for p in (source/'thermal'/split).glob('*.jpg')}
        if not rgb or set(rgb)!=set(ir):raise ValueError('RGB/IR split names differ or empty: '+split)
        paired[split]=sorted(rgb)
        for name in paired[split]:
            for path in (rgb[name],ir[name]):
                with Image.open(path) as image:
                    if image.size!=(640,480):raise ValueError('Current full-SH Probe requires 640x480 images: '+str(path))
                hashes[str(path)]=file_sha(path)
    if set(paired['train'])&set(paired['test']):raise ValueError('Train/test frame overlap')
    opt=SimpleNamespace(phys_material_mode='sam3_depth',phys_material_config=str(config),phys_material_manifest=manifest)
    material,config_path,config_hash,document,manifest_path=resolve_material_inputs(SimpleNamespace(source_path=str(source)),opt)
    if set(document['frames'])!=set(paired['train']):raise ValueError('Require complete train-only SAM3 mask coverage')
    hashes[str(manifest_path)]=file_sha(manifest_path)
    pixel_counts=np.zeros(3,np.int64);metal_frames=0
    for name in paired['train']:
        record=document['frames'][name]
        labels,scores=_frame_data(source,manifest_path.parent,record,480,640,'cpu',name)
        pixel_counts+=np.bincount(labels.ravel(),minlength=3);metal_frames+=int((labels==2).any())
        for key in ('labels','scores'):hashes[str(manifest_path.parent/record[key])]=record[key+'_sha256']
    reference_response=None;reference_palette=None;palette_length=None;shape=None;raw_checks=[]
    for split in ('train','test'):
        for name in paired[split]:
            path=source/'raw_images'/(name+'.jpg')
            dn,palette,meta=extract_flir(path,byte_order);hashes[str(path)]=file_sha(path)
            response=meta['planck'];palette_hash=meta['palette_sha256']
            if reference_response is None:
                reference_response=response;reference_palette=palette_hash;shape=meta['shape'];palette_length=len(palette)
            if response!=reference_response or len(palette)!=palette_length or meta['shape']!=shape:
                raise ValueError('Current Probe requires a shared response/native grid and equal palette lengths')
            raw_checks.append(dict(name=name,role=split,meta=meta))
    # Diagnose byte order only on training frames. Do not fit or inspect test colors.
    sample_names=[paired['train'][i] for i in sorted({0,len(paired['train'])//2,len(paired['train'])-1})]
    ordering=[]
    for name in sample_names:
        metrics={}
        for order in ('swap','native'):
            dn,_,meta=extract_flir(source/'raw_images'/(name+'.jpg'),order)
            u=(dn.astype(float)-meta['median'])/meta['span']+.5
            metrics[order]=dict(in_display_span_fraction=float(((u>=0)&(u<=1)).mean()),
                neighbor_jump_over_span=float(np.median(np.abs(np.diff(dn.astype(float),axis=1)))/meta['span']))
        other='native' if byte_order=='swap' else 'swap'
        if (metrics[byte_order]['neighbor_jump_over_span']>20*max(metrics[other]['neighbor_jump_over_span'],1e-8)
                and metrics[other]['in_display_span_fraction']>metrics[byte_order]['in_display_span_fraction']+.2):
            raise ValueError('Selected PNG byte order is inconsistent; use --png-byte-order '+other)
        ordering.append(dict(name=name,candidates=metrics))
    fit=[n for j,n in enumerate(paired['train']) if j%5!=0];validation=[n for j,n in enumerate(paired['train']) if j%5==0]
    if min(len(fit),len(validation))<2:raise ValueError('Need at least 2 fit and 2 validation training views')
    return dict(status='passed',scene=str(source),material_config=str(config_path),material_config_sha256=config_hash,
        mask_manifest=str(manifest_path),train=paired['train'],test=paired['test'],
        expected_fit_frames=fit,expected_validation_frames=validation,mask_pixel_counts=pixel_counts.tolist(),
        metal_mask_frames=metal_frames,raw_shape=shape,camera_response=reference_response,palette_sha256=reference_palette,
        palette_mode='known_per_frame_FLIR_metadata',palette_count=len({r['meta']['palette_sha256'] for r in raw_checks}),
        png_byte_order=byte_order,byte_order_checks=ordering,raw_checks=raw_checks,
        input_hashes=hashes,actual_optimizer_updates=0)


def check_hashes(records):
    changed=[p for p,h in records.items() if not Path(p).is_file() or file_sha(p)!=h]
    if changed:raise RuntimeError('Selected scene inputs changed: '+', '.join(changed[:8]))


def validate_mapping(path,audit,threads):
    """Frozen adapter provenance and production input checks, no optimizer steps."""
    import torch
    from tools.fit_flir_display_mapping import load_model
    d=json.loads(path.read_text())
    if (d.get('kind')!='approximate_FLIR_display_adapter' or resolve(d['settings']['scene'])!=Path(audit['scene'])
            or d.get('order')!='colorize_then_resize' or d.get('test_used_for_fitting_or_selection') is not False
            or d.get('png_byte_order')!=audit['png_byte_order']):
        raise ValueError('Mapping must belong to this scene and supported fit-only protocol')
    fit=set(d['fit_frames']);val=set(d['validation_frames'])
    if fit&val or fit|val!=set(audit['train']) or (fit|val)&set(audit['test']):
        raise ValueError('Mapping fit/validation/test split provenance mismatch')
    mapping,_=load_model(path,'cpu')
    if any(not torch.isfinite(p).all() for p in mapping.parameters()):raise ValueError('Nonfinite mapping')
    # The original adapter writes input hashes alongside mapping.json.
    provenance=path.parent/'manifest.json'
    if not provenance.is_file():raise ValueError('Mapping input manifest is missing')
    saved=json.loads(provenance.read_text())
    if {r['name'] for r in saved}!=set(audit['train']):raise ValueError('Mapping provenance does not cover all train frames')
    for record in saved:
        for role in ('raw','target'):
            current=audit['input_hashes'].get(record[role+'_path'])
            if current!=record[role+'_sha256']:raise ValueError('Mapping input hash mismatch: '+record['name'])
    print('[MAPPING VERIFIED] fit={} validation={} test_for_fit=0 raw={} -> 640x480'.format(
        len(fit),len(val),audit['raw_shape']),flush=True)
    return d


def training_command(args,source,config,output,mapping):
    command=[sys.executable,'-u',str(ROOT/'train.py'),'-s',str(source),'-m',str(output),
        '--data_branch','rgbt','--eval','--load2gpu_on_the_fly','--iterations',str(args.iterations),
        '--phys_material_mode','sam3_depth','--phys_material_config',str(config),
        '--phys_probe_mode','full_sh_radiometric_attributes','--phys_attribute_mapping',str(mapping),
        '--phys_aux_start_iter','20000','--phys_aux_ramp_iters','5000','--phys_aux_mse_max_weight','0.05',
        '--phys_probe_every','1000','--phys_decouple_steps','128',
        '--ir_late_consistency_weight','0.2','--late_lr_schedule','cosine',
        '--late_lr_start_iter','20000','--late_lr_end_iter','30000','--late_lr_final_factor','1.0']
    if args.material_manifest:command+=['--phys_material_manifest',str(resolve(args.material_manifest))]
    return command


class CheckpointController:
    def before_loop(self,L):return None
    def after_step(self,iteration,L):
        if iteration==BOUNDARY:save_complete_checkpoint(L,iteration)


def train_worker(plan_path):
    plan=json.loads(Path(plan_path).read_text());os.chdir(ROOT)
    import torch
    torch.set_num_threads(plan['threads']);torch.set_num_interop_threads(1)
    if torch.cuda.device_count()!=1:raise RuntimeError('Expected one selected CUDA device')
    check_hashes(plan['input_hashes']);check_hashes(plan['source_hashes'])
    source=(ROOT/'train.py').read_text();tree,inventory=transform_training(source)
    write_json(Path(plan['output'])/'main_loop_state_inventory.json',dict(carried_locals=inventory,
        hook='after ordinary optimizer steps; at 20000 only',loss_and_optimizers_unchanged=True))
    sys.argv=plan['train_command'][2:]
    namespace=dict(__name__='__main__',__file__=str(ROOT/'train.py'),__shared=CheckpointController())
    # The AST only adds the audited checkpoint callback. No experiment optimizer
    # adapter, alternative loss, Probe scheduling or source-file edits are used.
    exec(compile(tree,str(ROOT/'train.py'),'exec'),namespace)
    output=Path(plan['output'])
    audit=json.loads((output/'complete_state_20000.json').read_text())
    if audit['status']!='passed' or not audit['live_state_unchanged']:
        raise RuntimeError('Complete 20k checkpoint failed verification')
    cache=torch.load(output/'point_cloud'/('iteration_%d'%plan['iterations'])/'radiometric_probe.pt',map_location='cpu')
    if not all(cache.get(k) for k in ('main_loss_feedback','IR_SH_feedback_verified','slope_learnable')):
        raise RuntimeError('Final learnable-slope Probe/main feedback was not verified')
    write_json(output/'main_training_complete.json',dict(status='complete',iterations=plan['iterations'],
        complete_20k_checkpoint=audit['checkpoint'],main_feedback_verified=True,slope_learnable=True))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-s','--scene',help='Scene folder, or scene name under data/RGBT-Scenes')
    parser.add_argument('--material-config',help='Matching SAM3 configuration for the selected scene')
    parser.add_argument('--material-manifest',default='')
    parser.add_argument('-m','--model-path',help='Fresh model output; default output/<scene>_full_sam3_<timestamp>')
    parser.add_argument('--gpu',type=int,default=0)
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--iterations',type=int,default=30000)
    parser.add_argument('--mapping',help='Reuse an already fitted adapter for THIS scene; otherwise fit first')
    parser.add_argument('--mapping-steps',type=int,default=800)
    parser.add_argument('--png-byte-order',choices=['swap','native'],default='swap')
    parser.add_argument('--check-only',action='store_true',help='Full read-only data/mask/metadata checks; no fitting/training')
    parser.add_argument('--dry-run',action='store_true',help='Print commands after lightweight path/config checks only')
    parser.add_argument('--self-check',action='store_true',help='CPU complete-state codec and AST tests; no optimizer updates')
    parser.add_argument('--check-report',help='Optional JSON destination for read-only checks')
    parser.add_argument('--_worker-plan',help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args._worker_plan:return train_worker(args._worker_plan)
    if args.self_check:
        from tools.full_training_state import self_check
        return self_check(resolve(args.check_report or 'output/code_audit_scene_launcher/state_self_check.json'))
    if not args.scene or not args.material_config:parser.error('--scene and --material-config are required')
    if args.threads<1 or args.iterations<=BOUNDARY or args.mapping_steps<1:parser.error('Invalid threads/iterations/mapping budget')
    source=scene_path(args.scene);config=resolve(args.material_config)
    if not source.is_dir() or not config.is_file():raise FileNotFoundError('Scene or material config does not exist')
    from utils.material_config import load_material_config
    load_material_config(config)
    output=resolve(args.model_path) if args.model_path else ROOT/'output'/(source.name+'_full_sam3_'+time.strftime('%Y%m%d_%H%M%S'))
    if output==source or source in output.parents or output in source.parents:raise ValueError('Output must be separate from dataset')
    mapping=resolve(args.mapping) if args.mapping else output/'dn_mapping/mapping.json'
    fit_command=[sys.executable,'-u',str(ROOT/'tools/fit_flir_display_mapping.py'),'fit',
        '--scene',str(source),'--output',str(mapping.parent),'--model','monotone','--order','colorize_then_resize',
        '--geometry','affine','--steps',str(args.mapping_steps),'--pixels','2048','--val-every','5',
        '--png-byte-order',args.png_byte_order,'--device','cpu','--threads',str(args.threads)]
    train_command=training_command(args,source,config,output,mapping)
    if args.dry_run:
        print('[DRY RUN] no output/data checks/fitting/training started')
        if not args.mapping:print('[MAPPING] '+shlex.join(fit_command))
        print('[TRAIN] '+shlex.join(train_command));print('[CHECKPOINT] '+str(output/'complete_state_20000.pt'));return
    audit=audit_scene(source,config,str(resolve(args.material_manifest)) if args.material_manifest else '',args.png_byte_order,args.threads)
    print('[INPUTS VERIFIED] scene={} train={} test={} material_pixels={} native_raw={}'.format(
        source.name,len(audit['train']),len(audit['test']),audit['mask_pixel_counts'],audit['raw_shape']),flush=True)
    if args.mapping:validate_mapping(mapping,audit,args.threads)
    if args.check_only:
        if args.check_report:write_json(resolve(args.check_report),audit)
        print('[CHECK ONLY PASSED] actual optimizer updates=0; no mapping fit or 3DGS training');return
    gpu=gpu_selection(args.gpu)
    if output.exists():raise FileExistsError('Choose a fresh model output: '+str(output))
    output.mkdir(parents=True)
    write_json(output/'scene_input_audit.json',audit)
    if not args.mapping:checked_run(fit_command)
    mapping_doc=validate_mapping(mapping,audit,args.threads)
    # Exercise the production preflight before allocating a full training model.
    from arguments import ModelParams,OptimizationParams,PipelineParams
    from utils.attribute_probe_pipeline import validate_attribute_inputs
    from utils.full_sh_probe import configure_full_sh_probe
    p=argparse.ArgumentParser();mp=ModelParams(p,None);op=OptimizationParams(p);PipelineParams(p)
    parsed=p.parse_args(train_command[3:]);dataset=mp.extract(parsed);opt=op.extract(parsed)
    configure_full_sh_probe(opt);validate_attribute_inputs(dataset,opt)
    selected_inputs=dict(audit['input_hashes']);selected_inputs[str(mapping)]=file_sha(mapping)
    selected_inputs[str(mapping.parent/'manifest.json')]=file_sha(mapping.parent/'manifest.json')
    # Core training modules and the two runtime tools only; offline SAM3 jobs are
    # not dependencies. The actual selected mask/config/data files ARE guarded.
    sources=[ROOT/'train.py',ROOT/'render.py',Path(__file__).resolve(),ROOT/'tools/full_training_state.py',
        ROOT/'tools/fit_flir_display_mapping.py',ROOT/'tools/probe_full_sh_attributes.py']
    for directory in ('arguments','scene','utils','gaussian_renderer'):
        sources.extend((ROOT/directory).rglob('*.py'))
    source_hashes={str(p):file_sha(p) for p in sorted(set(sources)) if '__pycache__' not in p.parts}
    plan=dict(scene=str(source),output=str(output),iterations=args.iterations,threads=args.threads,gpu=gpu,
        input_hashes=selected_inputs,source_hashes=source_hashes,train_command=train_command,
        mapping=str(mapping),mapping_fit_frames=mapping_doc['fit_frames'],mapping_validation_frames=mapping_doc['validation_frames'],
        full_checkpoint_schema='physir_complete_training_boundary_v2',checkpoint_iteration=BOUNDARY,
        profile='unchanged production full-SH + learnable slope + true-IR-GT feedback; no VarPro/normalized-prior experiment')
    plan_path=output/'main_launch.json';write_json(plan_path,plan)
    env=os.environ.copy();env.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=str(gpu['cuda_ordinal']),
        OMP_NUM_THREADS=str(args.threads),MKL_NUM_THREADS=str(args.threads),OPENBLAS_NUM_THREADS=str(args.threads),
        NUMEXPR_NUM_THREADS=str(args.threads),PYTHONHASHSEED='0',PYTHONUNBUFFERED='1')
    # Recheck the device after CPU adapter fitting, which may take a few minutes.
    gpu_selection(args.gpu)
    checked_run([sys.executable,'-u',str(Path(__file__).resolve()),'--_worker-plan',str(plan_path)],env)
    check_hashes(selected_inputs);check_hashes(source_hashes)
    print('COMPLETED: '+str(output),flush=True)


if __name__=='__main__':main()
