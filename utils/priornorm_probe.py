"""PriorNorm bounded emissivity regression used by the RoadBlock C profile."""
import json
import math
from pathlib import Path
import time
import types
from tools.full_training_state import (digest, rng_state, tensor_hash, dump_wrapper,
    WRAPPERS, file_sha as sha, write_json)
ROOT=Path(__file__).resolve().parents[1]
BOUNDARY=20000
END=30000


def set_metal_start(field, start):
    """Change the starting epsilon only; preserve T, priors, bounds and slopes."""
    import torch
    low, high = field.low, field.high
    metal = torch.isclose(low, low.new_tensor(.2), atol=1e-6) & torch.isclose(high, high.new_tensor(.4), atol=1e-6)
    if not metal.any() or not torch.allclose(field.initial_eps[metal], low.new_full((int(metal.sum()),), .3), atol=1e-6):
        raise ValueError('Expected metal prior=.3 and fixed bounds=.2..4')
    before = tensor_hash({'temperature_raw': field.temperature_raw})
    prior = tensor_hash({'initial_eps': field.initial_eps, 'low': low, 'high': high})
    other = tensor_hash({'other_epsilon_raw': field.epsilon_raw[~metal]})
    with torch.no_grad():
        fraction = (float(start)-low[metal])/(high[metal]-low[metal])
        if not ((fraction > 0) & (fraction < 1)).all():
            raise ValueError('Metal start must be strictly inside .2..4')
        field.epsilon_raw[metal] = torch.logit(fraction)
    assert before == tensor_hash({'temperature_raw': field.temperature_raw})
    assert prior == tensor_hash({'initial_eps': field.initial_eps, 'low': low, 'high': high})
    assert other == tensor_hash({'other_epsilon_raw': field.epsilon_raw[~metal]})
    values = field.values('learn_eps_temperature_directional', 0)
    if not torch.allclose(values['cluster_emissivity'][metal], low.new_full((int(metal.sum()),), start), atol=1e-6):
        raise RuntimeError('Physical initial epsilon differs from requested start')
    controls = dict(metal_start=float(start), metal_prior_center=.3, metal_bounds=[.2,.4],
                    temperature_initialization_metal_reference=.3,
                    temperature_initialization='fit-only native-DN adjoint, unchanged reference epsilon',
                    temperature_raw_sha256=before, prior_and_bounds_sha256=prior,
                    nonmetal_start_unchanged=True, slope_start=0., metal_regions=int(metal.sum()))
    field.o.experiment_initialization_controls = controls
    return controls


def freeze_hash(field):
    return tensor_hash({name:getattr(field,name) for name in ('temperature_raw','environment_raw','direction_raw')})


def save_attribute_phase(live,iteration):
    import torch
    from tools.probe_full_sh_attributes import TEMPERATURE_VARIANT
    field=live.problem.field
    with torch.no_grad():
        v=field.values(TEMPERATURE_VARIANT,live.last_step)
        metal=(live.problem.labels==2)&field.supported
        regions=torch.unique(field.ids[metal],sorted=True)
        row=dict(iteration=iteration,attribute_step=live.last_step,
            phase=live._eps_first['phase'],updates=live.updates,
            metal_epsilon0_median=float(v['emissivity_ref'][metal].median()),
            metal_slope_per_K_median=float(v['emissivity_slope_per_K'][metal].median()),
            metal_T_median_C=float(v['temperature'][metal].median())-273.15,
            environment_K=float(v['environment_K']),metal_region_ids=regions.cpu().tolist(),
            metal_region_epsilon0=v['cluster_emissivity'][regions].cpu().tolist(),
            metal_region_slope_per_K=v['cluster_slope_per_K'][regions].cpu().tolist(),
            requires_grad={n:bool(getattr(field,n).requires_grad) for n in
                ('epsilon_raw','slope_raw','temperature_raw','environment_raw','direction_raw')})
    with (live.log_path.parent/'phase_fields.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
    print('[EPS-FIRST FIELD] '+json.dumps(row),flush=True)


def varpro_flags(field,joint):
    field.epsilon_raw.requires_grad_(False)
    field.slope_raw.requires_grad_(False)
    for name in ('temperature_raw','environment_raw','direction_raw'):
        getattr(field,name).requires_grad_(joint)


def endpoint_values(field, original_values, u, variant, step):
    values=original_values(variant,step)
    endpoints=field.low[:,None]+(field.high-field.low)[:,None]*u
    span=field.o.temp_max-field.o.temp_min
    fraction=(field.o.temperature_ref-field.o.temp_min)/span
    eps0=(1-fraction)*endpoints[:,0]+fraction*endpoints[:,1]
    slope=(endpoints[:,1]-endpoints[:,0])/span
    temp=values['temperature']
    values.update(emissivity=eps0[field.ids]+slope[field.ids]*(temp-field.o.temperature_ref),
        cluster_emissivity=eps0,emissivity_ref=eps0[field.ids],
        emissivity_slope_per_K=slope[field.ids],cluster_slope_per_K=slope,
        cluster_emissivity_low=endpoints[:,0],cluster_emissivity_high=endpoints[:,1])
    return values


def prepare_varpro_views(live,names,g,scene,geo,pipe,background,iteration):
    import torch
    import torch.nn.functional as F
    from train import compute_geo_fields
    p=live.problem;lookup={v.image_name:v for v in scene.getTrainCameras()};records=[]
    try:
        for name in names:
            if name not in p.roles['fit'] or name not in p.frames:
                raise RuntimeError('Conditional solve attempted to use a non-fit camera')
            view=lookup[name];old=view.world_view_transform.device
            record=dict(view=view,old_device=old,frame=p.frames[name]);records.append(record)
            view.load2device(g.get_xyz.device)
            with torch.no_grad():
                deform=compute_geo_fields(view,g,geo,pipe,background,live.opt,iteration,
                    lambda _:0.,1.,g.get_xyz.device,live.dataset.is_blender,
                    feature_set='thermal',eval_mode=True)
                deform=tuple(x.detach() if torch.is_tensor(x) else x for x in deform)
                means=g.get_xyz if pipe.convert_SHs_python else g.get_xyz+deform[0]
                direction=F.normalize(means-view.camera_center[None],dim=1)
                basis=((direction[p.field.metal_indices].double()-p.direction_mean)/p.direction_std).float()
            record.update(deform=deform,basis=basis)
        return records
    except BaseException:
        release_varpro_views(records)
        raise


def release_varpro_views(records):
    for record in records:
        record['view'].load2device(record['old_device'])


def varpro_data(live,record,g,pipe,background,values,step):
    import torch
    import torch.nn.functional as F
    from gaussian_renderer import render
    from tools.probe_full_sh_attributes import TEMPERATURE_VARIANT
    p=live.problem;field=p.field;frame=record['frame']
    signal=field.signal(values,record['basis'],TEMPERATURE_VARIANT,step)
    colors=torch.stack([signal/p.signal_scale,torch.ones_like(signal),(p.labels==2).to(signal)],-1).contiguous()
    image=render(record['view'],g,pipe,torch.zeros_like(background),*record['deform'],g.get_xyz.device,
        is_6dof=live.dataset.is_6dof,feature_set='thermal',override_color=colors,
        detach_geometry=True,disable_atmosphere=True)['render']
    native=F.grid_sample(image[None],p.grid,align_corners=False,padding_mode='zeros')[0]
    acc=native[1];q=native[0]*p.signal_scale+(1-acc)*values['environment_signal']
    valid=p.fov & torch.isfinite(acc) & (acc.detach()>=p.options.min_accumulation)
    if not valid.any():raise RuntimeError('No supported DN pixels: '+record['view'].image_name)
    error=(q-frame['target'])/p.loss_scale
    dn=F.smooth_l1_loss(error[valid],torch.zeros_like(error[valid]),beta=p.options.huber_beta)
    core=valid & (native[2].detach()/acc.detach().clamp_min(1e-8)>.9)
    metal=(F.smooth_l1_loss(error[core],torch.zeros_like(error[core]),beta=p.options.huber_beta)
           if core.sum()>=4 else q.new_zeros(()))
    normalized=(q-p.response.c['O']-frame['meta']['median'])/frame['meta']['span']+.5
    display=p.display(normalized[None],frame['xy'],(480,640),palette=frame.get('display_palette'))[0];mask=frame['display_valid']
    sh=(F.smooth_l1_loss(display[mask],frame['teacher'][mask],beta=.02)
        if mask.any() else q.new_zeros(()))
    loss=dn+p.options.metal_weight*metal+p.options.sh_weight*sh
    return loss,dict(DN_Huber=dn,metal_Huber=metal,SH_display_Huber=sh)


def varpro_objective(live,records,g,pipe,background,step,active_values):
    """Exact first derivative of the existing full fit-view objective, no steps.

    Evaluate/backpropagate one view at a time. Do not retain render graphs,
    populate live .grad buffers, or differentiate through main geometry/SH.
    """
    import numpy as np
    import torch
    from tools.probe_full_sh_attributes import TEMPERATURE_VARIANT
    s=live._varpro;field=live.problem.field
    flags={name:p.requires_grad for name,p in field.named_parameters()}
    for p in field.parameters():p.requires_grad_(False)
    leaf=torch.tensor(np.asarray(active_values),device=field.temperature_raw.device,dtype=torch.float32,requires_grad=True)
    gradient=torch.zeros_like(leaf);data_gradient=torch.zeros_like(leaf)
    objective=0.;components=dict(DN_Huber=0.,metal_Huber=0.,SH_display_Huber=0.)
    try:
        for record in records:
            # Fresh index_copy graph per frame: no retain_graph across views.
            full=s['u'].flatten().index_copy(0,s['active_index'],leaf).reshape_as(s['u'])
            s['trial_u']=full
            values=field.values(TEMPERATURE_VARIANT,step)
            data,parts=varpro_data(live,record,g,pipe,background,values,step)
            scaled=data/len(records)
            grad=torch.autograd.grad(scaled,leaf)[0]
            gradient.add_(grad.detach());data_gradient.add_(grad.detach())
            objective+=float(scaled.detach())
            for key in components:components[key]+=float(parts[key].detach())/len(records)
            s['trial_u']=None
        full=s['u'].flatten().index_copy(0,s['active_index'],leaf).reshape_as(s['u']);s['trial_u']=full
        values=field.values(TEMPERATURE_VARIANT,step)
        prior=field.prior(values,TEMPERATURE_VARIANT,step)
        gradient.add_(torch.autograd.grad(prior,leaf)[0].detach())
        objective+=float(prior.detach());components['prior']=float(prior.detach())
        components['material_prior_terms']=detached_prior_terms(field,values,TEMPERATURE_VARIANT,step,live._material_area_prior)
        gradient=gradient.cpu().double().numpy();data_gradient=data_gradient.cpu().double().numpy()
        if not math.isfinite(objective) or not np.isfinite(gradient).all():
            raise FloatingPointError('Nonfinite bounded conditional objective/gradient')
        return objective,gradient,components,data_gradient
    finally:
        s['trial_u']=None
        for name,p in field.named_parameters():p.requires_grad_(flags[name])


def projected_endpoint_gradient(x,gradient,margin):
    import numpy as np
    return x-np.clip(x-gradient,margin,1-margin)


def sync_endpoint_compatibility(live,u):
    import torch
    field=live.problem.field;s=live._varpro
    if s['read_only']:raise RuntimeError('Endpoint parameter updates forbidden in restoration checks')
    with torch.no_grad():
        logits=torch.logit(u)
        field.epsilon_raw.copy_(.5*(logits[:,0]+logits[:,1]))
        field.slope_raw.copy_(logits[:,1]-logits[:,0])
        s['u']=u.detach().clone()


def solve_varpro(live,iteration,g,scene,geo,pipe,background):
    import numpy as np
    import torch
    from scipy.optimize import minimize
    s=live._varpro;settings=s['options'];step=iteration-live.start
    if s['read_only']:raise RuntimeError('Conditional optimization forbidden in restoration check')
    before_rng=digest(rng_state());start=time.monotonic()
    names=sorted(live.problem.roles['fit'])
    if len(names)!=49:raise RuntimeError('This comparison requires the same 49 Probe fit cameras')
    records=prepare_varpro_views(live,names,g,scene,geo,pipe,background,iteration)
    x0=s['u'].flatten()[s['active_index']].cpu().double().numpy().copy()
    margin=settings['margin'];evaluations=0
    print('[VARPRO INNER START] iteration={} views=49 variables={} maxiter={}'.format(iteration,len(x0),settings['maxiter']),flush=True)
    try:
        f0,g0,parts0,dg=varpro_objective(live,records,g,pipe,background,step,x0)
        best=dict(x=x0.copy(),value=f0,gradient=g0.copy(),parts=parts0)
        last=dict(x=x0.copy(),value=f0,gradient=g0.copy())
        def objective(x):
            nonlocal evaluations
            evaluations+=1
            if evaluations==1 and np.array_equal(x,x0):value,gradient,parts=f0,g0,parts0
            else:value,gradient,parts,_=varpro_objective(live,records,g,pipe,background,step,x)
            if value<best['value']:
                best.update(x=np.asarray(x).copy(),value=value,gradient=gradient.copy(),parts=parts)
            last.update(x=np.asarray(x).copy(),value=value,gradient=gradient.copy())
            if evaluations%5==0:
                print('[VARPRO INNER EVAL] iteration={} eval={} objective={:.9g}'.format(iteration,evaluations,value),flush=True)
            return value,gradient
        result=minimize(objective,x0,jac=True,method='L-BFGS-B',bounds=[(margin,1-margin)]*len(x0),
            options=dict(maxiter=settings['maxiter'],maxfun=settings['maxfun'],
                ftol=settings['ftol'],gtol=settings['gtol'],maxls=12,maxcor=10))
        # Commit only the best feasible candidate evaluated on the UNCHANGED
        # exact objective. Failed/iteration-limited solves are reported honestly.
        if not np.array_equal(last['x'],best['x']):
            value,gradient,parts,_=varpro_objective(live,records,g,pipe,background,step,best['x'])
            best.update(value=value,gradient=gradient,parts=parts)
        accepted=bool(best['value']<f0 and np.isfinite(best['x']).all()
                      and np.all(best['x']>=margin) and np.all(best['x']<=1-margin))
        if accepted:
            candidate=s['u'].flatten().clone()
            candidate[s['active_index']]=torch.tensor(best['x'],device=candidate.device,dtype=candidate.dtype)
            sync_endpoint_compatibility(live,candidate.reshape_as(s['u']))
            s['commits']+=1
        projected=projected_endpoint_gradient(best['x'],best['gradient'],margin)
        projected_norm=float(np.max(np.abs(projected)))
        row=dict(iteration=iteration,fit_views=49,objective_before=f0,objective_after=best['value'],
            objective_nonincreasing=bool(best['value']<=f0),accepted=accepted,
            solver_success=bool(result.success),solver_status=int(result.status),message=str(result.message),
            optimizer_iterations=int(result.nit),function_evaluations=evaluations,
            projected_gradient_inf=projected_norm,
            stationarity_tolerance_met=projected_norm<=settings['gtol'],
            elapsed_seconds=time.monotonic()-start,before=parts0,after=best['parts'],
            full_objective=True,global_optimum_claimed=False)
        s['solver_calls']+=1;s['last_inner_iteration']=iteration;s['last_inner']=row
        with (live.log_path.parent/'varpro_inner.jsonl').open('a') as out:out.write(json.dumps(row,allow_nan=False)+'\n')
        print('[VARPRO INNER END] '+json.dumps(row),flush=True)
        return accepted
    finally:
        release_varpro_views(records)
        if digest(rng_state())!=before_rng:raise RuntimeError('Conditional solve changed main RNG state')


def audit_varpro(live,L,plan):
    """Real renderer derivatives and objective parity; absolutely no solver calls."""
    import numpy as np
    import torch
    from tools.probe_full_sh_attributes import TEMPERATURE_VARIANT
    s=live._varpro;field=live.problem.field;g=L['gaussians']
    names=sorted(live.problem.roles['fit'])
    records=prepare_varpro_views(live,names,g,L['scene'],L['geo_model'],L['pipe'],L['background'],BOUNDARY+1)
    flags={name:p.requires_grad for name,p in field.named_parameters()}
    before=tensor_hash(field.state_dict());rng_before=digest(rng_state())
    try:
        x=s['u'].flatten()[s['active_index']].cpu().double().numpy()
        f,grad,parts,dg=varpro_objective(live,records,g,L['pipe'],L['background'],1,x)
        full_data=np.zeros(s['u'].numel());full_data[s['active_index'].cpu().numpy()]=dg
        full_data=full_data.reshape(-1,2)
        center=float(np.linalg.norm(full_data[:,0]+full_data[:,1]))
        difference=float(np.linalg.norm(.5*(full_data[:,1]-full_data[:,0])))
        if not all(math.isfinite(a) and a>0 for a in (center,difference)):
            raise RuntimeError('Conditional DATA has no emissivity/slope derivative')
        region_class=np.zeros(len(field.low),dtype=np.int64)
        region_class[field.ids.cpu().numpy()]=live.problem.labels.cpu().numpy()
        class_gradients={}
        for label,name in ((1,'nonmetal'),(2,'metal')):
            selected=full_data[region_class==label]
            common=float(np.linalg.norm(selected[:,0]+selected[:,1]))
            different=float(np.linalg.norm(.5*(selected[:,1]-selected[:,0])))
            if not all(math.isfinite(a) and a>0 for a in (common,different)):
                raise RuntimeError('No usable conditional emissivity/slope DATA derivative for '+name)
            class_gradients[name]=dict(common_endpoint_direction=common,difference_endpoint_direction=different)
        # Finite differences are observation evaluations, never parameter commits.
        coordinate=int(np.argmax(np.abs(grad)));delta=min(.001,
            .25*(x[coordinate]-s['options']['margin']),.25*(1-s['options']['margin']-x[coordinate]))
        if delta<=0:raise RuntimeError('No interior coordinate for finite-difference check')
        plus=x.copy();minus=x.copy();plus[coordinate]+=delta;minus[coordinate]-=delta
        fp=varpro_objective(live,records,g,L['pipe'],L['background'],1,plus)[0]
        fm=varpro_objective(live,records,g,L['pipe'],L['background'],1,minus)[0]
        finite_difference=(fp-fm)/(2*delta)
        relative_error=abs(finite_difference-grad[coordinate])/max(abs(finite_difference),abs(grad[coordinate]),1e-7)
        if relative_error>.05:raise RuntimeError('Full-fit objective derivative differs from finite differences: '+str(relative_error))
        # Original field function and endpoint function must agree before any solve.
        with torch.no_grad():
            new=field.values(TEMPERATURE_VARIANT,1);old=s['original_values'](TEMPERATURE_VARIANT,1)
            parity=max(float((new[k]-old[k]).abs().max()) for k in
                ('temperature','emissivity','emissivity_ref','emissivity_slope_per_K'))
        if parity>2e-6:raise RuntimeError('Endpoint forward model differs from the restored model')
        # A genuine data VJP in the physical endpoint coordinates, plus checks
        # that no main model parameter participates in the Probe data graph.
        u=s['u'].detach().clone().requires_grad_(True);s['trial_u']=u
        for name,p in field.named_parameters():p.requires_grad_(True)
        values=field.values(TEMPERATURE_VARIANT,2501)
        data,_=varpro_data(live,records[0],g,L['pipe'],L['background'],values,2501)
        sources=(g._thermal_features_dc,g._thermal_features_rest,g._xyz,g._scaling,g._rotation,g._opacity)
        leaked=torch.autograd.grad(data,sources,retain_graph=True,allow_unused=True)
        if any(v is not None and bool((v!=0).any()) for v in leaked):
            raise RuntimeError('Conditional Probe leaked gradients to main Gaussian/SH')
        derivatives=torch.autograd.grad(data,(u,field.temperature_raw,field.environment_raw),allow_unused=True)
        norms=[float(v.norm()) if v is not None else 0. for v in derivatives]
        if not all(math.isfinite(v) and v>0 for v in norms):raise RuntimeError('Missing endpoint/T/environment data derivative')
        live.verified.update(('emissivity','slope'));live.probe_path_verified=True
        s['audit']=dict(status='passed',actual_optimizer_updates=0,conditional_solver_calls=0,
            endpoint_forward_max_error=parity,probe_objective=f,
            data_endpoint_common_gradient_norm=center,data_endpoint_difference_gradient_norm=difference,
            class_data_gradients=class_gradients,
            full_objective_finite_difference=dict(coordinate=coordinate,analytic=float(grad[coordinate]),
                numerical=finite_difference,relative_error=relative_error,fit_views=len(records),live_parameters_updated=False),
            endpoint_T_environment_data_gradient_norms=norms,
            main_Gaussian_SH_data_gradients_zero=True,
            endpoint_T_chain_rule_preserved=True,fit_views_only=names,
            temperature_initialization_unchanged=True,compatibility_logits_not_optimized_by_Adam=True)
        audit_material_prior(live,grad,dg)
        s['audit']['material_prior_check']=live._material_area_prior['gradient_audit']
        write_json(live.log_path.parent/'varpro_gradient_check.json',s['audit'])
        print('[VARPRO AUDIT] '+json.dumps(s['audit']),flush=True)
    finally:
        s['trial_u']=None
        for name,p in field.named_parameters():p.requires_grad_(flags[name])
        release_varpro_views(records)
    if tensor_hash(field.state_dict())!=before or digest(rng_state())!=rng_before:
        raise RuntimeError('Derivative audit changed field parameters or RNG')


def save_varpro_diagnostics(live,iteration):
    import torch
    from tools.probe_full_sh_attributes import TEMPERATURE_VARIANT
    s=live._varpro;field=live.problem.field
    with torch.no_grad():
        values=field.values(TEMPERATURE_VARIANT,live.last_step);classes={}
        for label,name in ((1,'nonmetal'),(2,'metal')):
            mask=(live.problem.labels==label)&field.supported;T=values['temperature'][mask]-273.15
            classes[name]=dict(points=int(mask.sum()),temperature_C_min_max=[float(T.min()),float(T.max())],
                temperature_C_p05_p50_p95=torch.quantile(T,T.new_tensor([.05,.5,.95])).cpu().tolist(),
                epsilon0_median=float(values['emissivity_ref'][mask].median()),
                epsilon_median=float(values['emissivity'][mask].median()),
                slope_per_K_median=float(values['emissivity_slope_per_K'][mask].median()),
                below_zero_points=int((T<0).sum()),above_40C_points=int((T>40).sum()))
    row=dict(iteration=iteration,probe_update_events=live.updates,fit_visits=s['fit_visits'],
        solver_calls=s['solver_calls'],endpoint_commits=s['commits'],outer_Adam_updates=s['outer_updates'],
        classes=classes,last_inner=s['last_inner'],
        material_prior_terms=detached_prior_terms(field,values,TEMPERATURE_VARIANT,live.last_step,live._material_area_prior))
    with (live.log_path.parent/'varpro_fields.jsonl').open('a') as out:out.write(json.dumps(row,allow_nan=False)+'\n')
    print('[VARPRO FIELDS] '+json.dumps(row),flush=True)


def material_area_weights(region_area, region_class, active, uniform_mix):
    """Fixed class-normalized area weights, with a small uniform prior floor."""
    import torch
    if not 0 <= uniform_mix <= 1 or region_area.ndim != 1:
        raise ValueError('Invalid prior area normalization')
    if not torch.isfinite(region_area).all() or bool((region_area < 0).any()):
        raise ValueError('Nonfinite/negative region visibility area')
    weights=torch.zeros_like(region_area)
    for label in (1,2):
        selected=active & (region_class==label)
        count=int(selected.sum())
        if not count:continue
        area=region_area[selected];total=area.sum()
        proportional=area/total if float(total)>0 else torch.full_like(area,1/count)
        weights[selected]=(1-uniform_mix)*proportional+uniform_mix/count
    return weights.detach()


def normalized_prior_terms(field, values, variant, step, state):
    """Change only known-material epsilon/slope aggregation, not other priors."""
    import torch
    import torch.nn.functional as F
    active=field.active_regions;classes=state['region_class'];weights=state['weights']
    width=field.high-field.low
    epsilon_penalty=F.smooth_l1_loss((values['cluster_emissivity']-field.initial_eps)/width,
        torch.zeros_like(width),beta=.1,reduction='none')
    slope_penalty=(values['cluster_slope_per_K']*(field.o.temp_max-field.o.temp_min)/width).square()
    # Evaluate the ORIGINAL environment/direction priors exactly, with its
    # epsilon/slope terms zero. No subtractive cancellation or detached T path.
    neutral=dict(values,cluster_emissivity=field.initial_eps,
                 cluster_slope_per_K=torch.zeros_like(values['cluster_slope_per_K']))
    remaining=state['original_prior'](neutral,variant,step)
    terms={'environment_direction_unchanged':remaining}
    for label,name in ((1,'nonmetal'),(2,'metal')):
        selected=active & (classes==label);scale=state['options'][name+'_scale']
        terms[name+'_epsilon']=field.o.emissivity_prior_weight*scale*(weights[selected]*epsilon_penalty[selected]).sum()
        terms[name+'_slope']=(field.o.slope_prior_weight*scale*(weights[selected]*slope_penalty[selected]).sum()
            if field.slope_active(variant,step) else remaining.new_zeros(()))
    # Unknown class keeps its original contribution: SUM / ALL active regions.
    unknown=active & (classes==0);denominator=active.sum().clamp_min(1)
    terms['unknown_epsilon_unchanged']=field.o.emissivity_prior_weight*epsilon_penalty[unknown].sum()/denominator
    terms['unknown_slope_unchanged']=(field.o.slope_prior_weight*slope_penalty[unknown].sum()/denominator
        if field.slope_active(variant,step) else remaining.new_zeros(()))
    return terms


def detached_prior_terms(field,values,variant,step,state):
    return {k:float(v.detach()) for k,v in normalized_prior_terms(field,values,variant,step,state).items()}


def install_material_area_prior(live,L,plan):
    """49-fit-view alpha adjoint at restored 20k; no optimizer updates."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    from gaussian_renderer import render
    from utils.full_sh_probe import _preserve_rng
    p=live.problem;field=p.field;g=L['gaussians'];n=len(field.low)
    if sha(ROOT/'tools/probe_full_sh_attributes.py')!='9c57706f011c5d600cac3d1c6fb496f52a394ae477f9f9be3120903f9ec654c1':
        raise RuntimeError('Probe prior source changed; review normalization adapter')
    original_prior=field.prior;options=plan['prior_normalization']
    before=tensor_hash(field.state_dict());rng_before=digest(rng_state())
    main_before=digest({k:dump_wrapper(L[k]) for k in WRAPPERS})
    supported=field.supported.bool();ids=field.ids;labels=p.labels
    counts=torch.stack([torch.bincount(ids[supported & (labels==c)],minlength=n) for c in (0,1,2)])
    if bool(((counts>0).sum(0)>1).any()):raise RuntimeError('A region mixes material classes')
    classes=counts.argmax(0).long();classes[~field.active_regions]=-1
    if not torch.equal(counts.sum(0)>0,field.active_regions):raise RuntimeError('Active-region ownership differs')
    names=sorted(p.roles['fit'])
    if len(names)!=49 or set(names)&(set(p.roles['validation'])|set(p.roles['test'])):
        raise RuntimeError('Area weights require exactly the disjoint 49 fit views')
    point_area=torch.zeros_like(field.temperature_raw,dtype=torch.float64)
    view_audit=[];records=[]
    # Preserve main sampling streams; use only geometry, FOV and accumulation.
    # Targets, teacher colors, validation and test data never enter these weights.
    with _preserve_rng(g.get_xyz.device):
        try:
            records=prepare_varpro_views(live,names,g,L['scene'],L['geo_model'],L['pipe'],L['background'],BOUNDARY)
            for number,record in enumerate(records,1):
                colors=torch.ones((len(ids),3),device=ids.device,dtype=torch.float32,requires_grad=True)
                image=render(record['view'],g,L['pipe'],torch.zeros_like(L['background']),*record['deform'],g.get_xyz.device,
                    is_6dof=live.dataset.is_6dof,feature_set='thermal',override_color=colors,
                    detach_geometry=True,disable_atmosphere=True)['render']
                native=F.grid_sample(image[None],p.grid,align_corners=False,padding_mode='zeros')[0]
                acc=native[1].detach()
                valid=p.fov & torch.isfinite(acc) & (acc>=p.options.min_accumulation)
                if not bool(valid.any()):raise RuntimeError('No DN support for area weights: '+record['view'].image_name)
                score=native[0][valid].sum()
                contribution=torch.autograd.grad(score,colors)[0][:,0].detach()
                if not torch.isfinite(contribution).all() or float(contribution.min()) < -1e-6:
                    raise RuntimeError('Invalid alpha-adjoint visibility weights')
                error=abs(float(contribution.sum())-float(score.detach()))/max(float(score.detach()),1e-8)
                if error>2e-4:raise RuntimeError('Visibility VJP mass check failed: '+str(error))
                point_area.add_(contribution.clamp_min(0).double())
                view_audit.append(dict(name=record['view'].image_name,valid_pixels=int(valid.sum()),
                    alpha_mass=float(contribution.sum()),relative_adjoint_mass_error=error))
                del contribution,score,native,image,colors
                if number==1 or number%12==0 or number==len(records):
                    print('[PRIOR AREA] fit views {}/49; no targets/validation/test or parameter updates'.format(number),flush=True)
        finally:release_varpro_views(records)
    point_area[~supported]=0
    region_area=torch.zeros(n,device=ids.device,dtype=torch.float64).scatter_add_(0,ids,point_area)
    weights=material_area_weights(region_area,classes,field.active_regions,options['uniform_mix']).to(field.low)
    state=dict(options=options,region_class=classes.detach(),weights=weights.detach(),original_prior=original_prior,
        point_area=point_area.detach(),region_area=region_area.detach())
    live._material_area_prior=state
    def prior(self,values,variant,step):
        return sum(normalized_prior_terms(self,values,variant,step,state).values())
    field.prior=types.MethodType(prior,field)
    summary=dict(status='passed',kind='fit_visible_area_class_normalized_epsilon_slope_priors',
        reference_iteration=BOUNDARY,weights_frozen_after_initialization=True,fit_view_names=names,
        validation_and_test_used=False,observed_DN_or_teacher_values_used=False,options=options,
        unknown_prior='exact original SUM / all active regions',
        environment_direction_priors_unchanged=True,material_prior_centers_and_bounds_unchanged=True,
        temperature_initialization_unchanged=True,weak_visibility_uniform_weight_fraction=options['uniform_mix'],
        coefficient_interpretation='per-class weighted mean; not comparable to old all-region mean coefficient',
        classes={},view_checks=view_audit,actual_optimizer_updates=0,conditional_solver_calls=0)
    for label,name in ((0,'unknown'),(1,'nonmetal'),(2,'metal')):
        selected=field.active_regions & (classes==label);rids=selected.nonzero().flatten()
        if label and len(rids) and abs(float(weights[selected].sum())-1)>2e-6:
            raise RuntimeError('Class prior weights do not sum to one: '+name)
        order=rids[torch.argsort(region_area[rids],descending=True)[:8]]
        scale=options.get(name+'_scale')
        summary['classes'][name]=dict(regions=int(selected.sum()),supported_points=int(counts[label].sum()),
            effective_pixel_area=float(region_area[selected].sum()),weight_sum=float(weights[selected].sum()),
            epsilon_coefficient=field.o.emissivity_prior_weight*scale if scale is not None else None,
            slope_coefficient=field.o.slope_prior_weight*scale if scale is not None else None,
            top_regions=[dict(region=int(r),points=int(counts[label,r]),area=float(region_area[r]),
                normalized_weight=float(weights[r]),original_global_mean_weight=1/int(field.active_regions.sum())) for r in order])
    path=live.log_path.parent/'material_prior_weights.npz'
    np.savez_compressed(path,region_area=region_area.cpu().numpy(),point_area=point_area.cpu().numpy(),
        region_class=classes.cpu().numpy(),weights=weights.cpu().numpy(),active=field.active_regions.cpu().numpy(),
        material_ids=ids.cpu().numpy(),supported=supported.cpu().numpy(),fit_view_names=np.asarray(names))
    summary['weights_sha256']=sha(path)
    if any(t.requires_grad for t in (weights,region_area,point_area)):
        raise RuntimeError('Prior weights must not hold autograd graphs')
    if tensor_hash(field.state_dict())!=before or digest(rng_state())!=rng_before:
        raise RuntimeError('Weight construction changed field parameters or main RNG')
    if digest({k:dump_wrapper(L[k]) for k in WRAPPERS})!=main_before:
        raise RuntimeError('Weight construction changed main parameters/Adam')
    summary.update(field_and_main_Adam_unchanged=True,main_RNG_unchanged=True,no_retained_autograd_graph=True)
    state['audit']=summary
    p.options.experiment_prior_normalization=summary
    write_json(live.log_path.parent/'material_prior_normalization.json',summary)
    print('[MATERIAL PRIOR] '+json.dumps({k:v for k,v in summary.items() if k not in ('view_checks','fit_view_names')}),flush=True)


def audit_material_prior(live,full_gradient,data_gradient):
    """Log prior/data gradient scales and verify derivatives without updates."""
    import numpy as np
    import torch
    from tools.probe_full_sh_attributes import TEMPERATURE_VARIANT
    s=live._varpro;state=live._material_area_prior;field=live.problem.field
    def expanded(g):
        result=np.zeros(s['u'].numel());result[s['active_index'].cpu().numpy()]=g
        return result.reshape(-1,2)
    prior=expanded(full_gradient-data_gradient);data=expanded(data_gradient);classes=state['region_class'].cpu().numpy()
    ratios={}
    for label,name in ((1,'nonmetal'),(2,'metal')):
        a=prior[classes==label];b=data[classes==label]
        ratios[name]={}
        for kind,pa,pb in [('epsilon_common',a[:,0]+a[:,1],b[:,0]+b[:,1]),
                ('slope_difference',.5*(a[:,1]-a[:,0]),.5*(b[:,1]-b[:,0]))]:
            pn=float(np.linalg.norm(pa));dn=float(np.linalg.norm(pb))
            ratios[name][kind]=dict(prior_norm=pn,data_norm=dn,prior_over_data=pn/dn if dn else None)
    # Perturb scratch endpoint leaves only: exercise nonzero slope derivatives.
    # The live parameters and committed endpoints are not changed.
    before=tensor_hash(field.state_dict());committed=tensor_hash({'endpoint_u':s['u']});rng_before=digest(rng_state())
    trial=s['u'].detach().double().clone();trial[:,0]=.35;trial[:,1]=.65;trial.requires_grad_(True)
    low=field.low.double();width=(field.high-field.low).double();ends=low[:,None]+width[:,None]*trial
    values=field.values(TEMPERATURE_VARIANT,1)
    values=dict(values,cluster_emissivity=.6*ends[:,0]+.4*ends[:,1],
                cluster_slope_per_K=(ends[:,1]-ends[:,0])/(field.o.temp_max-field.o.temp_min))
    parts=normalized_prior_terms(field,values,TEMPERATURE_VARIANT,1,state)
    slope=parts['metal_slope']+parts['nonmetal_slope']
    gradient=torch.autograd.grad(slope,trial)[0]
    if not torch.isfinite(gradient).all() or float(gradient.norm())<=0:
        raise RuntimeError('Nonzero slope has no normalized prior derivative')
    if tensor_hash(field.state_dict())!=before or tensor_hash({'endpoint_u':s['u']})!=committed or digest(rng_state())!=rng_before:
        raise RuntimeError('Prior derivative check changed live parameters/endpoints/RNG')
    report=dict(status='passed',initial_prior_data_gradient_scales=ratios,
        nonzero_scratch_slope_prior_gradient_norm=float(gradient.norm()),scratch_endpoints_only=True,
        no_main_or_Probe_parameter_updates=True,actual_optimizer_updates=0,conditional_solver_calls=0)
    state['gradient_audit']=report
    write_json(live.log_path.parent/'material_prior_gradient_check.json',report)
    print('[PRIOR GRADIENT AUDIT] '+json.dumps(report),flush=True)


def install_varpro(L,plan):
    import torch
    from tools.probe_full_sh_attributes import TEMPERATURE_VARIANT
    live=L['live_feedback'];p=live.problem;field=p.field
    if live.updates or live.last_step:raise RuntimeError('Require the same unstepped 20k Probe')
    settings=plan['varpro'];original_values=field.values
    with torch.no_grad():
        values=original_values(TEMPERATURE_VARIANT,0)
        ends=torch.stack((values['cluster_emissivity_low'],values['cluster_emissivity_high']),1)
        u=((ends-field.low[:,None])/(field.high-field.low)[:,None]).detach().clone()
    active_index=field.active_regions[:,None].expand(-1,2).reshape(-1).nonzero().flatten()
    s=dict(u=u,active_index=active_index,trial_u=None,original_values=original_values,
        options=settings,read_only=plan['role']=='restore_check',solver_calls=0,commits=0,
        fit_visits=0,outer_updates=0,last_inner_iteration=-100000,last_inner=None,audit={})
    live._varpro=s
    def stages(self,variant,step):return True,int(step)>plan['phase1_steps']
    def slope_active(self,variant,step):return True
    field.stages=types.MethodType(stages,field);field.slope_active=types.MethodType(slope_active,field)
    def values_function(self,variant,step):
        return endpoint_values(self,original_values,s['u'] if s['trial_u'] is None else s['trial_u'],variant,step)
    field.values=types.MethodType(values_function,field)
    install_material_area_prior(live,L,plan)
    # Preserve original Adam hyperparameters/state for T/environment/direction.
    # This checkpoint's Probe is unstepped; endpoint moments are not repurposed.
    previous=live.optimizer;groups=[]
    for idx in (0,3,4):groups.append(dict(previous.param_groups[idx]))
    live.optimizer=torch.optim.Adam(groups,**previous.defaults)
    for group in live.optimizer.param_groups:
        for parameter in group['params']:
            if parameter in previous.state:live.optimizer.state[parameter]=previous.state[parameter]
    schedule=dict(kind='epsilon_slope_first_online',phase1_main_steps=plan['phase1_steps'],
        epsilon_start_iteration=20001,slope_start_iteration=20001,
        temperature_start_iteration=20001+plan['phase1_steps'],environment_start_iteration=20001+plan['phase1_steps'],
        direction_start_iteration=20001+plan['phase1_steps'],
        epsilon_and_slope_remain_trainable_in_phase2=True,
        emissivity_solver='periodic bounded normalized physical endpoints; no endpoint Adam',
        data_loss='unchanged DN + metal core + SH display + original priors',
        main_loss_feedback='unchanged detached-confidence true-IR-GT MSE',
        metal_prior_center=.3,metal_bounds=[.2,.4],temperature_initialization_reference_epsilon=.3)
    p.options.experiment_optimization_schedule=schedule
    p.options.experiment_varpro=dict(settings,kind='bounded_inexact_online_variable_projection',views=49,
        fit_view_names=sorted(p.roles['fit']),validation_and_test_used_for_inner=False,
        solver='L-BFGS-B, exact first derivatives of complete existing objective',
        projected_variables='normalized physical emissivity endpoints, regional epsilon0 and slope retained',
        exact_global_projection=False,temperature_spatial_weight=0.,
        physical_temperature_ground_truth=False,main_feedback_target='observed_IR_GT')
    live._eps_first=dict(phase1_steps=plan['phase1_steps'],initial_frozen_hash=freeze_hash(field),
        frozen_verified=False,phase1_updates=0,phase2_updates=0,phase=None)
    varpro_flags(field,False)
    def update(self,iteration,view,g,scene,geo,pipe,background):
        from tools.probe_full_sh_attributes import TEMPERATURE_VARIANT
        step=iteration-self.start;self.last_step=step;state=self._eps_first
        joint=step>plan['phase1_steps']
        if joint and not state['frozen_verified']:
            if freeze_hash(field)!=state['initial_frozen_hash']:
                raise RuntimeError('T/environment/direction changed during phase 1')
            state['frozen_verified']=True
        varpro_flags(field,joint)
        phase='joint' if joint else 'epsilon_slope_only'
        if state['phase']!=phase:
            print('[VARPRO PHASE] iteration={} phase={}'.format(iteration,phase),flush=True);state['phase']=phase
        stats=dict(probe_phase=phase,conditional_solver='bounded_periodic',T_spatial_weight=0.)
        if view.image_name in p.frames:
            s['fit_visits']+=1;committed=False
            due=(iteration-s['last_inner_iteration']>=settings['every']
                 or iteration==BOUNDARY+plan['phase1_steps']+1 or iteration==END)
            if due and not s['read_only']:
                committed=solve_varpro(self,iteration,g,scene,geo,pipe,background)
            records=prepare_varpro_views(self,[view.image_name],g,scene,geo,pipe,background,iteration)
            try:
                self.optimizer.zero_grad(set_to_none=True)
                values=field.values(TEMPERATURE_VARIANT,step)
                data,parts=varpro_data(self,records[0],g,pipe,background,values,step)
                prior=field.prior(values,TEMPERATURE_VARIANT,step)
                if joint and not s['read_only']:
                    active={'temperature':field.temperature_raw,'environment':field.environment_raw}
                    unchecked={k:v for k,v in active.items() if k not in self.verified}
                    if unchecked:
                        gradients=torch.autograd.grad(data,tuple(unchecked.values()),retain_graph=True,allow_unused=True)
                        for (key,_),grad in zip(unchecked.items(),gradients):
                            norm=float(grad.norm()) if grad is not None else 0.
                            if not math.isfinite(norm) or norm<=0:raise RuntimeError('Missing outer DATA gradient: '+key)
                            self.verified.add(key)
                    if not torch.isfinite(data+prior):raise FloatingPointError('Nonfinite outer objective')
                    (data+prior).backward()
                    if field.temperature_raw.grad is not None:field.temperature_raw.grad[~field.supported]=0
                    for parameter in field.parameters():
                        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                            raise FloatingPointError('Nonfinite outer gradient')
                    self.optimizer.step();self.optimizer.zero_grad(set_to_none=True)
                    s['outer_updates']+=1
                if (joint and not s['read_only']) or committed:
                    self.updates+=1
                    state['phase2_updates' if joint else 'phase1_updates']+=1
                stats.update({k:float(v.detach()) for k,v in parts.items()},prior=float(prior.detach()),
                    inner_solver_calls=s['solver_calls'],endpoint_commits=s['commits'],
                    outer_Adam_updates=s['outer_updates'],fit_visits=s['fit_visits'])
            finally:release_varpro_views(records)
        else:stats['Probe_update_skipped_nonfit_view']=True
        if iteration%1000==0 or iteration in (20001,BOUNDARY+plan['phase1_steps'],BOUNDARY+plan['phase1_steps']+1,END):
            write_json(self.log_path.parent/'phase_audit.json',dict(iteration=iteration,schedule=schedule,
                phase1_frozen_T_environment_direction_verified=state['frozen_verified'],
                phase1_attribute_updates=state['phase1_updates'],phase2_attribute_updates=state['phase2_updates'],
                inner_solver_calls=s['solver_calls'],endpoint_commits=s['commits'],outer_Adam_updates=s['outer_updates']))
        return stats
    live._update=types.MethodType(update,live)
    original_save=live.save
    def save(self,iteration):
        original_save(iteration)
        if iteration>END:return
        target=self.log_path.parent/('varpro_state_%d.pt'%iteration)
        torch.save(dict(endpoint_u=s['u'].cpu(),active_index=s['active_index'].cpu(),
            inner_solver_calls=s['solver_calls'],endpoint_commits=s['commits'],
            outer_Adam_updates=s['outer_updates'],last_inner=s['last_inner'],options=settings),target)
    live.save=types.MethodType(save,live)
    write_json(live.log_path.parent/'epsilon_first_schedule.json',schedule)
    write_json(live.log_path.parent/'varpro_options.json',p.options.experiment_varpro)
    audit_varpro(live,L,plan)
    save_varpro_diagnostics(live,BOUNDARY)
