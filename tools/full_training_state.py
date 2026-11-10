"""Lossless main-training boundary snapshots, including deferred Probe state.

Extracted from the verified complete-state codec; no training entry point.
"""
import ast
import gc
import hashlib
import json
from pathlib import Path
import pickle
import random
import shutil
import time

SCHEMA='physir_complete_training_boundary_v2'
BOUNDARY=20000

def dump_feedback(live):
    if live is None:raise RuntimeError('This profile requires live Probe feedback')
    initialized=live.problem is not None
    if initialized != (live.optimizer is not None):raise RuntimeError('Inconsistent Probe initialization')
    return dict(initialized=initialized,problem=live.problem,optimizer=dump_optimizer(live.optimizer),
        updates=live.updates,last_step=live.last_step,verified=live.verified,sh_verified=live.sh_verified,
        probe_path_verified=live.probe_path_verified)


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def save_complete_checkpoint(L,iteration):
    """Capture after step, keep production's pending Probe/assets untouched."""
    import torch
    if iteration!=BOUNDARY:raise ValueError('Expected completed iteration 20000')
    target=Path(L['model_path'])/'complete_state_20000.pt'
    if target.exists():raise FileExistsError('Preserve existing complete checkpoint: '+str(target))
    torch.cuda.synchronize() if torch.cuda.is_initialized() else None
    before_rng=digest(rng_state());state=capture_state(L,iteration)
    expected=semantic_parts(state,portable=True)
    if not state['wrappers']['gaussians']['optimizer']['state_dict']['state']:
        raise RuntimeError('Missing main Adam moments at the checkpoint boundary')
    # Snapshot the ordinary pre-step saved assets rather than refreshing them.
    # This is exactly what the uninitialized production Probe will read next.
    root=target.parent;assets=root/'complete_state_20000_assets';assets.mkdir(exist_ok=False)
    selected=[]
    for name in ('input.ply','cameras.json','cfg_args','result.txt'):
        source=root/name
        if source.is_file():shutil.copy2(source,assets/name);selected.append(assets/name)
    for name in ('point_cloud','Geometry','GeometryRGB','Atmos','Planck','Radiometric','phys_probe'):
        source=root/name/'iteration_20000'
        if source.exists():
            destination=assets/name/source.name;destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copytree(source,destination);selected.extend(p for p in destination.rglob('*') if p.is_file())
    required=[assets/'point_cloud/iteration_20000/point_cloud.ply',
              assets/'Geometry/iteration_20000/geometry.pth',assets/'GeometryRGB/iteration_20000/geometry.pth',
              assets/'phys_probe/iteration_20000/full_sh_observations.pt']
    if not all(p.is_file() for p in required):raise RuntimeError('Required 20k Probe assets were not saved')
    asset_hashes={str(p.relative_to(root)):file_sha(p) for p in sorted(selected)}
    state['asset_snapshot']=dict(root=str(assets),sha256=asset_hashes,
        semantics='unaltered production pre-step saves; full model/Adam below is after-step')
    tmp=target.with_suffix('.pt.tmp');torch.save(state,tmp);tmp.replace(target)
    # Verify on CPU to avoid allocating a second model on a nearly full GPU.
    loaded=torch.load(target,map_location='cpu')
    if semantic_parts(loaded,portable=True)!=expected:raise RuntimeError('Serialized complete-state mismatch')
    del loaded
    current=semantic_parts(capture_state(L,iteration),portable=True)
    if current!=expected or digest(rng_state())!=before_rng:
        raise RuntimeError('Saving the checkpoint changed live model/Adam/Probe/camera/RNG state')
    report=dict(status='passed',schema=SCHEMA,iteration=iteration,
        boundary='after_all_optimizer_updates_and_zero_grad',checkpoint=str(target),checkpoint_sha256=file_sha(target),
        parts=expected,portable_serialization_roundtrip_verified=True,live_state_unchanged=True,
        main_Adam_moments_saved=True,Probe_initialized=state['feedback']['initialized'],
        Probe_updates=state['feedback']['updates'],Probe_not_eagerly_initialized=True,
        ordinary_20k_assets_preserved=True,assets=asset_hashes,
        optimizer_state_entries={k:len(v['optimizer']['state_dict']['state']) for k,v in state['wrappers'].items() if v is not None},
        note='V2 preserves a genuinely deferred Probe; continuation must use this codec and asset snapshot, not weights-only resume.')
    write_json(root/'complete_state_20000.json',report)
    del state;gc.collect()
    print('[COMPLETE CHECKPOINT] '+str(target)+' | main Adam/RNG/cameras/DropGaussian/Probe state saved; continuing training',flush=True)

def tensor_hash(named):
    h = hashlib.sha256()
    for key, t in sorted(named.items()):
        h.update(json.dumps([key, str(t.dtype), list(t.shape)]).encode())
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def callable_descriptor(value):
    import marshal
    result={'type':type(value).__module__+'.'+type(value).__qualname__,
            'module':getattr(value,'__module__',None),'name':getattr(value,'__qualname__',getattr(value,'__name__',None))}
    code=getattr(value,'__code__',None)
    if code is not None:
        result['code_sha256']=hashlib.sha256(marshal.dumps(code)).hexdigest()
        result['constant_state_sha256']=digest((getattr(value,'__defaults__',None),getattr(value,'__kwdefaults__',None),
            tuple(c.cell_contents for c in (getattr(value,'__closure__',None) or ()))))
    return result


def tensor_record(value):
    import torch
    return dict(data=value.detach(), requires_grad=bool(value.requires_grad),
                grad=None if value.grad is None else value.grad.detach())


def restore_parameter(target, record):
    import torch
    with torch.no_grad():target.data=record['data']
    target.requires_grad_(record['requires_grad'])
    target.grad=record['grad']


HOOK_NAMES={'_backward_hooks','_backward_pre_hooks','_forward_hooks','_forward_pre_hooks',
            '_forward_hooks_with_kwargs','_forward_pre_hooks_with_kwargs','_state_dict_hooks',
            '_load_state_dict_pre_hooks','_load_state_dict_post_hooks','_state_dict_pre_hooks',
            '_forward_hooks_always_called'}


def dump_module(module):
    records={}
    for name,part in module.named_modules():
        if any(bool(getattr(part,k,{})) for k in HOOK_NAMES):
            raise RuntimeError('Unregistered module hooks require an explicit checkpoint codec: '+name)
        attrs={};functions={}
        for key,value in vars(part).items():
            if key in {'_parameters','_buffers','_modules'} or key in HOOK_NAMES:continue
            if callable(value):functions[key]=callable_descriptor(value)
            else:attrs[key]=value
        records[name]=dict(type=type(part).__module__+'.'+type(part).__qualname__,attrs=attrs,
            functions=functions,parameters={k:None if v is None else tensor_record(v) for k,v in part._parameters.items()},
            buffers=dict(part._buffers),children=list(part._modules))
    return records


def restore_module(module, records):
    current=dict(module.named_modules())
    if list(current)!=list(records):raise RuntimeError('Module graph differs from checkpoint')
    for name,state in records.items():
        part=current[name]
        if type(part).__module__+'.'+type(part).__qualname__!=state['type']:
            raise RuntimeError('Module type changed: '+name)
        if list(part._modules)!=state['children'] or set(part._parameters)!=set(state['parameters']):
            raise RuntimeError('Module structure changed: '+name)
        for key,desc in state['functions'].items():
            if callable_descriptor(getattr(part,key))!=desc:raise RuntimeError('Pure function changed: '+name+'.'+key)
        for key,value in state['attrs'].items():setattr(part,key,value)
        for key,record in state['parameters'].items():
            if record is None:
                if part._parameters[key] is not None:raise RuntimeError('Parameter None mismatch')
            else:restore_parameter(part._parameters[key],record)
        part._buffers=state['buffers']


def dump_optimizer(optimizer):
    if optimizer is None:return None
    # Include defaults and additional optimizer attributes, not only moments.
    extra={k:v for k,v in vars(optimizer).items() if k not in ('state','param_groups','defaults')}
    if any(callable(v) for v in extra.values()):raise RuntimeError('Unsupported callable optimizer attribute')
    return dict(type=type(optimizer).__module__+'.'+type(optimizer).__qualname__,
                state_dict=optimizer.state_dict(),defaults=optimizer.defaults,extra=extra)


def restore_optimizer(optimizer, state):
    if state is None:
        if optimizer is not None:raise RuntimeError('Unexpected optimizer')
        return
    if type(optimizer).__module__+'.'+type(optimizer).__qualname__!=state['type']:
        raise RuntimeError('Optimizer type differs')
    optimizer.defaults=state['defaults']
    optimizer.load_state_dict(state['state_dict'])
    for key,value in state['extra'].items():setattr(optimizer,key,value)


def dump_wrapper(obj):
    import torch
    if obj is None:return None
    attrs={};functions={};modules={};parameters={}
    for key,value in vars(obj).items():
        if key=='optimizer':continue
        if isinstance(value,torch.nn.Module):modules[key]=dump_module(value)
        elif isinstance(value,torch.nn.Parameter):parameters[key]=tensor_record(value)
        elif callable(value):functions[key]=callable_descriptor(value)
        else:attrs[key]=value
    return dict(type=type(obj).__module__+'.'+type(obj).__qualname__,attrs=attrs,functions=functions,
                modules=modules,parameters=parameters,optimizer=dump_optimizer(obj.optimizer))


def restore_wrapper(obj, state):
    if state is None:
        if obj is not None:raise RuntimeError('Unexpected model')
        return
    if type(obj).__module__+'.'+type(obj).__qualname__!=state['type']:
        raise RuntimeError('Model type changed')
    for key,desc in state['functions'].items():
        if callable_descriptor(getattr(obj,key))!=desc:raise RuntimeError('Reconstructed function changed: '+key)
    for key,value in state['attrs'].items():setattr(obj,key,value)
    for key,record in state['parameters'].items():restore_parameter(getattr(obj,key),record)
    for key,record in state['modules'].items():restore_module(getattr(obj,key),record)
    restore_optimizer(obj.optimizer,state['optimizer'])
    # Optimizer groups must still own the very same Parameter objects.
    owned={id(p) for key in state['parameters'] for p in [getattr(obj,key)]}
    owned.update(id(p) for key in state['modules'] for p in getattr(obj,key).parameters())
    if any(id(p) not in owned for group in obj.optimizer.param_groups for p in group['params']):
        raise RuntimeError('Optimizer Parameter identity was not preserved')


def digest(value, *, portable=False):
    """Stable semantic hash; handles aliases/cycles and every tensor byte."""
    import numpy as np
    import torch
    h=hashlib.sha256();seen={}
    def visit(x):
        if callable(x) and not isinstance(x,torch.nn.Module):
            if id(x) in seen:h.update(('ref:'+str(seen[id(x)])).encode());return
            seen[id(x)]=len(seen)
            import marshal
            h.update(repr((type(x).__name__,getattr(x,'__module__',None),getattr(x,'__qualname__',None))).encode())
            code=getattr(x,'__code__',None)
            if code is not None:h.update(marshal.dumps(code))
            visit(getattr(x,'__defaults__',None));visit(getattr(x,'__kwdefaults__',None))
            visit(tuple(c.cell_contents for c in (getattr(x,'__closure__',None) or ())))
            bound=getattr(x,'__self__',None)
            if bound is not None and not isinstance(bound,type):visit(bound)
            return
        if torch.is_tensor(x):
            h.update(repr(('tensor',str(x.dtype),tuple(x.shape),('portable' if portable else str(x.device)),bool(x.requires_grad))).encode())
            h.update(x.detach().cpu().contiguous().numpy().tobytes());return
        if isinstance(x,np.ndarray):
            h.update(repr(('ndarray',str(x.dtype),x.shape)).encode());h.update(x.tobytes());return
        if isinstance(x,(str,int,float,bool,type(None),bytes,torch.device)):
            h.update(repr((type(x).__name__,x)).encode());return
        if id(x) in seen:h.update(('ref:'+str(seen[id(x)])).encode());return
        seen[id(x)]=len(seen)
        if isinstance(x,dict):
            h.update(('dict:'+str(len(x))).encode())
            for key in sorted(x,key=lambda k:(type(k).__name__,str(k))):visit(key);visit(x[key])
        elif isinstance(x,(list,tuple,set)):
            h.update((type(x).__name__+':'+str(len(x))).encode())
            for item in sorted(x,key=repr) if isinstance(x,set) else x:visit(item)
        elif hasattr(x,'__dict__'):
            h.update((type(x).__module__+'.'+type(x).__qualname__).encode());visit(vars(x))
        else:h.update(pickle.dumps(x,protocol=4))
    visit(value);return h.hexdigest()


def rng_state():
    import numpy as np
    import torch
    return dict(python=random.getstate(),numpy=np.random.get_state(),cpu=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
        cudnn_benchmark=torch.backends.cudnn.benchmark,cudnn_deterministic=torch.backends.cudnn.deterministic,
        cudnn_tf32=torch.backends.cudnn.allow_tf32,matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        default_dtype=torch.get_default_dtype())


def restore_rng(state):
    import numpy as np
    import torch
    torch.backends.cudnn.benchmark=state['cudnn_benchmark']
    torch.backends.cudnn.deterministic=state['cudnn_deterministic']
    torch.backends.cudnn.allow_tf32=state['cudnn_tf32'];torch.backends.cuda.matmul.allow_tf32=state['matmul_tf32']
    torch.use_deterministic_algorithms(state['deterministic_algorithms'])
    torch.set_default_dtype(state['default_dtype'])
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['cpu'].cpu())
    if state['cuda']:torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


WRAPPERS=('gaussians','geo_model','rgb_geo_model','atmos_model','planck_model','radiometric_model')


SCALARS=('background','best_iteration','best_psnr','densify_initial_points','ema_loss_for_log',
         'fixed_time_interval','has_rgbt','late_l1_weight','prev_probe_point_count','start_iteration',
         'testing_iterations','saving_iterations')


CARRIED=set(WRAPPERS)|set(SCALARS)|{'dataset','device','ir_dropgaussian','iter_end','iter_start',
    'live_feedback','log_learning_rates','model_path','opt','path_atmosphere','pipe','progress_bar',
    'run_atmosphere_probe','run_material_probe','save_training_state','scene','smooth_term',
    'start_time','tb_writer','temporal_affine_model','temporal_pose_model','viewpoint_stack'}


def transform_training(source):
    tree=ast.parse(source);f=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_training_loop')
    loops=[n for n in f.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='iteration']
    if len(loops)!=1:raise RuntimeError('Training iteration loop changed')
    loop=loops[0];index=f.body.index(loop)
    class Names(ast.NodeVisitor):
        def __init__(self):self.store=set();self.load=set()
        def visit_Name(self,n):(self.store if isinstance(n.ctx,ast.Store) else self.load).add(n.id)
        def visit_FunctionDef(self,n):self.store.add(n.name)
        def visit_Lambda(self,n):pass
        def visit_ListComp(self,n):pass
        def visit_DictComp(self,n):pass
    before=Names();inside=Names()
    for node in f.body[:index]:before.visit(node)
    inside.visit(loop)
    carried=(before.store|{n.arg for n in f.args.args})&inside.load
    if carried!=(CARRIED-{'prev_probe_point_count'}):raise RuntimeError('Unclassified loop state: '+str(sorted(carried^(CARRIED-{'prev_probe_point_count'}))))
    end_with=loop.body[-1]
    if not isinstance(end_with,ast.With):raise RuntimeError('Training update boundary moved')
    call_names=[n.func.attr for n in ast.walk(end_with) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
    if call_names.count('step')!=8 or not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
            and n.func.attr=='update_learning_rate' and isinstance(n.func.value,ast.Name)
            and n.func.value.id=='radiometric_model' for n in ast.walk(end_with)):
        raise RuntimeError('Optimizer sequence changed; review the checkpoint boundary')
    assignments=['__shared_begin = 1','__shared_values = __shared.before_loop(locals())',
                 'if __shared_values is not None:','    __shared_begin = 20001']
    for key in SCALARS+('viewpoint_stack','start_time'):
        assignments.append('    '+key+' = __shared_values['+repr(key)+']')
    f.body[index:index]=ast.parse('\n'.join(assignments)).body
    loop.iter=ast.parse('range(__shared_begin, opt.iterations + 1)',mode='eval').body
    end_with.body.extend(ast.parse('__shared.after_step(iteration, locals())').body)
    ast.fix_missing_locations(tree)
    return tree,sorted(carried)


def capture_state(L, iteration):
    import torch
    if any(L[k] is not None for k in ('path_atmosphere','temporal_affine_model','temporal_pose_model')):
        raise RuntimeError('This experiment requires the current B configuration without optional extra models')
    g=L['gaussians'];scene=L['scene'];drop=L['ir_dropgaussian']
    scene_attrs={k:v for k,v in vars(scene).items() if k not in ('gaussians','model_path','load_model_path')}
    state=dict(schema=SCHEMA,iteration=iteration,boundary='after_all_optimizer_updates_and_zero_grad',
        config={k:vars(L[k]).copy() for k in ('dataset','opt','pipe')},
        wrappers={k:dump_wrapper(L[k]) for k in WRAPPERS},scene=scene_attrs,
        scalars={k:L[k] for k in SCALARS},
        camera_stack=None if L['viewpoint_stack'] is None else [v.image_name for v in L['viewpoint_stack']],
        train_camera_order=[v.image_name for v in scene.getTrainCameras()],
        test_camera_order=[v.image_name for v in scene.getTestCameras()],
        train_cameras_local_order=[v.image_name for v in L['train_cameras']],
        drop=dict(attrs={k:v for k,v in vars(drop).items() if k!='_generators'},
                  generators={str(k):v.get_state() for k,v in drop._generators.items()}),
        feedback=dump_feedback(L['live_feedback']),rng=rng_state(),
        elapsed=time.time()-L['start_time'],pure_smooth_function=callable_descriptor(L['smooth_term']))
    # Do not save the last iteration's autograd graph, timers, progress-bar or
    # writer handles: none supplies state to a later optimization update.
    for key in WRAPPERS:
        obj=L[key]
        if obj is not None:
            for group in obj.optimizer.param_groups:
                for p in group['params']:
                    if p.grad is not None and bool((p.grad!=0).any()):
                        raise RuntimeError('Checkpoint is not at a cleared-gradient boundary: '+key)
    return state


def semantic_parts(state, *, portable=False):
    return {k:digest(state[k],portable=portable) for k in ('wrappers','scene','scalars','camera_stack','train_camera_order',
        'test_camera_order','train_cameras_local_order','drop','feedback','rng','pure_smooth_function')}


def restore_state(L, state):
    import torch
    if state['schema'] not in (SCHEMA,'physir_complete_training_boundary_v1') or state['iteration']!=BOUNDARY:raise RuntimeError('Wrong full checkpoint')
    for label in ('dataset','opt','pipe'):
        current=vars(L[label]).copy();saved=state['config'][label].copy()
        if label=='dataset':current.pop('model_path',None);saved.pop('model_path',None)
        if current!=saved:raise RuntimeError('Continuation configuration differs: '+label)
    for name in WRAPPERS:restore_wrapper(L[name],state['wrappers'][name])
    scene=L['scene'];scene.__dict__.update(state['scene'])
    scene.gaussians=L['gaussians'];scene.model_path=L['model_path'];scene.load_model_path=L['model_path']
    train=scene.getTrainCameras();test=scene.getTestCameras()
    if [v.image_name for v in train]!=state['train_camera_order'] or [v.image_name for v in test]!=state['test_camera_order']:
        raise RuntimeError('Camera order differs after restore')
    L['train_cameras'][:]=train  # preserve the pre-created closure's list identity
    drop=L['ir_dropgaussian'];drop.__dict__.update(state['drop']['attrs']);drop._generators={}
    for dev,rs in state['drop']['generators'].items():
        gen=torch.Generator(device=torch.device(dev));gen.set_state(rs.cpu());drop._generators[torch.device(dev)]=gen
    live=L['live_feedback'];fb=state['feedback'];live.problem=fb['problem']
    live.log_path.parent.mkdir(parents=True,exist_ok=True)
    if fb.get('initialized',fb['problem'] is not None):
        field=live.problem.field;p=live.problem.options
        live.optimizer=torch.optim.Adam([
            dict(params=[field.temperature_raw],lr=p.temperature_lr),dict(params=[field.epsilon_raw],lr=p.emissivity_lr),
            dict(params=[field.slope_raw],lr=p.emissivity_lr*p.slope_lr_scale),
            dict(params=[field.environment_raw],lr=p.environment_lr),dict(params=[field.direction_raw],lr=p.direction_lr)])
        restore_optimizer(live.optimizer,fb['optimizer'])
    else:
        if fb['optimizer'] is not None:raise RuntimeError('Deferred Probe cannot have an optimizer')
        live.optimizer=None
    # V1 had an initialized Probe implicitly; normalize this metadata for audit.
    state=dict(state,feedback=dict(fb,initialized=fb.get('initialized',fb['problem'] is not None)))
    for key in ('updates','last_step','verified','sh_verified','probe_path_verified'):setattr(live,key,fb[key])
    lookup={v.image_name:v for v in train}
    values=dict(state['scalars']);values['viewpoint_stack']=(None if state['camera_stack'] is None
        else [lookup[n] for n in state['camera_stack']])
    values['start_time']=time.time()-state['elapsed']
    check=dict(L);check.update(values)
    restore_rng(state['rng'])
    actual=semantic_parts(capture_state(check,BOUNDARY));expected=semantic_parts(state)
    if actual!=expected:raise RuntimeError('Full-state roundtrip mismatch: '+str([k for k in actual if actual[k]!=expected[k]]))
    return values,actual


class _StateCheckScene:
    def getTrainCameras(self):return self.train
    def getTestCameras(self):return self.test


def _state_check_smooth(_):return 0.


def self_check(report_path):
    """CPU roundtrip/export of nonzero Adam state; no optimizer updates."""
    import numpy as np
    import torch
    from types import SimpleNamespace
    report_path=Path(report_path);report_path.parent.mkdir(parents=True,exist_ok=True)
    output=report_path.parent/('state_fixture_'+time.strftime('%Y%m%d_%H%M%S'))
    output.mkdir(exist_ok=False)
    old_rng=rng_state();original_step=torch.optim.Adam.step
    def forbidden(*args,**kwargs):raise RuntimeError('Optimizer updates forbidden in checkpoint self-check')
    torch.optim.Adam.step=forbidden
    try:
        def context(folder):
            result={}
            for name in WRAPPERS:
                p=torch.nn.Parameter(torch.arange(12.).reshape(4,3))
                optimizer=torch.optim.Adam([p],lr=.003)
                optimizer.state[p]=dict(step=torch.tensor(20000.),exp_avg=torch.full_like(p,.17),exp_avg_sq=torch.full_like(p,.31))
                result[name]=SimpleNamespace(weight=p,optimizer=optimizer,active_degree=3)
            folder.mkdir(parents=True,exist_ok=True)
            scene=_StateCheckScene();scene.train=[SimpleNamespace(image_name=n) for n in ('fit_a','fit_b')]
            scene.test=[SimpleNamespace(image_name='test_a')];scene.gaussians=result['gaussians']
            scene.model_path=scene.load_model_path=str(folder)
            cfg=SimpleNamespace(source_path='/synthetic',model_path=str(folder),eval=True,data_branch='rgbt')
            result.update(dataset=cfg,opt=SimpleNamespace(iterations=30000),pipe=SimpleNamespace(),scene=scene,
                ir_dropgaussian=SimpleNamespace(_generators={},seed=0),live_feedback=SimpleNamespace(problem=None,
                    optimizer=None,updates=0,last_step=0,verified=set(),sh_verified=False,probe_path_verified=False,
                    log_path=folder/'phys_probe/feedback/optimization.jsonl'),
                path_atmosphere=None,temporal_affine_model=None,temporal_pose_model=None,
                train_cameras=list(scene.train),viewpoint_stack=[scene.train[1]],model_path=str(folder),
                start_time=time.time()-100.,smooth_term=_state_check_smooth)
            result.update({name:0 for name in SCALARS})
            result.update(background=torch.zeros(3),has_rgbt=True,testing_iterations=[20000,30000],saving_iterations=[20000,30000])
            return result
        live=context(output/'source')
        # Exercise nested network state, its eval mode and optimizer identity.
        network=torch.nn.Sequential(torch.nn.Linear(2,2),torch.nn.Dropout(.1));network.eval()
        wrapped=SimpleNamespace(net=network,optimizer=torch.optim.Adam(network.parameters(),lr=.002))
        restored_network=torch.nn.Sequential(torch.nn.Linear(2,2),torch.nn.Dropout(.1))
        restored_wrapper=SimpleNamespace(net=restored_network,optimizer=torch.optim.Adam(restored_network.parameters(),lr=.7))
        saved_module=dump_wrapper(wrapped);restore_wrapper(restored_wrapper,saved_module)
        assert digest(dump_wrapper(restored_wrapper))==digest(saved_module)
        root=Path(live['model_path'])
        for rel in ('input.ply','cameras.json','cfg_args','result.txt','point_cloud/iteration_20000/point_cloud.ply',
                    'Geometry/iteration_20000/geometry.pth','GeometryRGB/iteration_20000/geometry.pth',
                    'phys_probe/iteration_20000/full_sh_observations.pt'):
            p=root/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'CPU state fixture only\n')
        before=rng_state();expected_draw=(random.random(),np.random.rand(),torch.rand(4));restore_rng(before)
        before_hash=semantic_parts(capture_state(live,BOUNDARY),portable=True)
        save_complete_checkpoint(live,BOUNDARY)
        actual_draw=(random.random(),np.random.rand(),torch.rand(4))
        assert expected_draw[:2]==actual_draw[:2] and torch.equal(expected_draw[2],actual_draw[2])
        assert live['live_feedback'].problem is None and live['live_feedback'].optimizer is None
        loaded=torch.load(root/'complete_state_20000.pt',map_location='cpu')
        assert semantic_parts(loaded,portable=True)==before_hash
        restored=context(output/'restored');values,parts=restore_state(restored,loaded)
        assert parts==semantic_parts(loaded)
        assert restored['live_feedback'].problem is None and restored['live_feedback'].optimizer is None
        assert float(restored['gaussians'].optimizer.state[restored['gaussians'].weight]['step'])==20000
        tree,inventory=transform_training((Path(__file__).resolve().parents[1]/'train.py').read_text())
        compile(tree,'train.py[checkpoint_hook]','exec')
        report=dict(status='passed',actual_training_steps=0,actual_optimizer_updates=0,
            nonzero_Adam_moments_and_step_restored=True,deferred_Probe_exactly_preserved=True,
            scene_camera_stack_and_model_modes_restored=True,RNG_unchanged_by_export=True,
            full_boundary_export_and_CPU_load_verified=True,training_AST_inventory_verified=True,
            state_parts=parts,carried_locals=inventory,fixture=str(output))
        write_json(report_path,report);print('[STATE SELF CHECK PASSED] '+str(report_path),flush=True)
        return report
    finally:
        torch.optim.Adam.step=original_step;restore_rng(old_rng)
