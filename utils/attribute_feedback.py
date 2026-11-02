"""Live epsilon(T) Probe and detached-confidence MSE against observed IR GT.

The DN/SH readout objective updates only the attribute optimizer. The feedback
objective updates the ordinary IR rendering, including its full SH coefficients.
Confidence is the inherited bounded heuristic, not calibrated uncertainty.
"""
import gc
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from gaussian_renderer import render
from utils.radiative_utils import radiative_confidence_values, radiative_confidence_mse_loss

LIVE_KIND = 'training_full_sh_radiometric_attribute_probe'


def enabled(opt):
    return (opt.phys_probe_mode == 'full_sh_radiometric_attributes'
            and float(opt.phys_aux_mse_max_weight) > 0.)


def validate(dataset, opt):
    if opt.phys_probe_mode == 'full_sh_radiometric_attributes' and (
            not math.isfinite(float(opt.phys_aux_mse_max_weight)) or opt.phys_aux_mse_max_weight < 0):
        raise ValueError('Attribute feedback weight must be finite and nonnegative')
    if not enabled(opt):
        return
    start = int(opt.phys_aux_start_iter)
    if (not dataset.eval or dataset.data_branch != 'rgbt' or dataset.is_6dof
            or start < opt.densify_until_iter or opt.iterations <= start
            or opt.phys_probe_every <= 0 or start % opt.phys_probe_every
            or opt.phys_aux_ramp_iters <= 0
            or not math.isfinite(opt.phys_aux_mse_max_weight)):
        raise ValueError('Attribute feedback requires additive RGBT --eval, fixed topology, '
                         'a Probe-aligned start, positive ramp and iterations beyond its start')
    if int(getattr(dataset, 'load_iteration', 0) or 0) > start:
        raise ValueError('Start fresh or at the feedback-start checkpoint; later resume '
                         'needs the corresponding live Probe/optimizer state')


class AttributeFeedback:
    def __init__(self, dataset, opt, model_path):
        self.dataset, self.opt = dataset, opt
        self.model = Path(model_path).resolve()
        self.start = int(opt.phys_aux_start_iter)
        self.problem = self.optimizer = None
        self.updates = 0
        self.last_step = 0
        self.verified = set()
        self.sh_verified = False
        self.probe_path_verified = False
        self.log_path = self.model/'phys_probe/feedback/optimization.jsonl'

    def _initialize(self, g):
        from tools.probe_full_sh_attributes import main as prepare
        from utils.attribute_probe_pipeline import _arguments
        from utils.full_sh_probe import _preserve_rng
        command = _arguments(self.model, self.start, self.opt)
        preparation = self.model.parent/(self.model.name+'_attribute_initialization')
        command += ['--steps', str(self.opt.iterations-self.start), '--output', str(preparation)]
        with _preserve_rng(g.get_xyz.device):
            self.problem = prepare(command, prepare_only=True)
        p, field = self.problem, self.problem.field
        if len(field.temperature_raw) != len(g.get_xyz):
            raise RuntimeError('Attribute point order/count differs from live fixed topology')
        self.optimizer = torch.optim.Adam([
            dict(params=[field.temperature_raw], lr=p.options.temperature_lr),
            dict(params=[field.epsilon_raw], lr=p.options.emissivity_lr),
            dict(params=[field.slope_raw], lr=p.options.emissivity_lr*p.options.slope_lr_scale),
            dict(params=[field.environment_raw], lr=p.options.environment_lr),
            dict(params=[field.direction_raw], lr=p.options.direction_lr)])
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        gc.collect()
        torch.cuda.empty_cache()
        print('[ATTRIBUTE FEEDBACK] live T/epsilon0/k; original readout objective; '
              'detached reliability weights; SH target=observed IR GT', flush=True)

    def _update(self, iteration, view, g, scene, geo, pipe, background):
        from tools.probe_full_sh_attributes import TEMPERATURE_VARIANT
        from train import compute_geo_fields
        p, field = self.problem, self.problem.field
        step = iteration-self.start
        self.last_step = step
        self.optimizer.zero_grad(set_to_none=True)
        if len(g.get_xyz) != len(field.temperature_raw):
            raise RuntimeError('Gaussian topology changed after Probe initialization')
        if view.image_name not in p.frames:
            return {}
        frame = p.frames[view.image_name]
        with torch.no_grad():
            deform = compute_geo_fields(view, g, geo, pipe, background, self.opt, iteration,
                lambda _:0., 1., g.get_xyz.device, self.dataset.is_blender,
                feature_set='thermal', eval_mode=True)
            means = g.get_xyz if pipe.convert_SHs_python else g.get_xyz+deform[0]
            directions = F.normalize(means-view.camera_center[None], dim=1)
            basis = ((directions[field.metal_indices].double()-p.direction_mean)/p.direction_std).float()
        values = field.values(TEMPERATURE_VARIANT, step)
        signal = field.signal(values, basis, TEMPERATURE_VARIANT, step)
        colors = torch.stack([signal/p.signal_scale, torch.ones_like(signal),
                              (p.labels == 2).to(signal)], -1).contiguous()
        image = render(view, g, pipe, torch.zeros_like(background), *deform, g.get_xyz.device,
                       is_6dof=self.dataset.is_6dof, feature_set='thermal',
                       override_color=colors, detach_geometry=True, disable_atmosphere=True)['render']
        native = F.grid_sample(image[None], p.grid, align_corners=False, padding_mode='zeros')[0]
        acc = native[1]
        q = native[0]*p.signal_scale+(1-acc)*values['environment_signal']
        valid = p.fov & torch.isfinite(acc) & (acc.detach() >= p.options.min_accumulation)
        if not valid.any():
            raise RuntimeError('No supported native DN pixels: '+view.image_name)
        error = (q-frame['target'])/p.loss_scale
        dn = F.smooth_l1_loss(error[valid], torch.zeros_like(error[valid]), beta=p.options.huber_beta)
        core = valid & (native[2].detach()/acc.detach().clamp_min(1e-8) > .9)
        metal = (F.smooth_l1_loss(error[core], torch.zeros_like(error[core]), beta=p.options.huber_beta)
                 if core.sum() >= 4 else q.new_zeros(()))
        normalized = (q-p.response.c['O']-frame['meta']['median'])/frame['meta']['span']+.5
        display_prediction = p.display(normalized[None], frame['xy'], (480, 640))[0]
        mask = frame['display_valid']
        sh = (F.smooth_l1_loss(display_prediction[mask], frame['teacher'][mask], beta=.02)
              if mask.any() else q.new_zeros(()))
        data = dn+p.options.metal_weight*metal+p.options.sh_weight*sh
        prior = field.prior(values, TEMPERATURE_VARIANT, step)
        if not torch.isfinite(data+prior):
            raise FloatingPointError('Nonfinite attribute objective')
        active = {'temperature':field.temperature_raw, 'environment':field.environment_raw}
        if field.stages(TEMPERATURE_VARIANT, step)[0]:
            active['emissivity'] = field.epsilon_raw
        if field.slope_active(TEMPERATURE_VARIANT, step):
            active['slope'] = field.slope_raw
        unchecked = {k:v for k,v in active.items() if k not in self.verified}
        if unchecked:
            gradients = torch.autograd.grad(data, tuple(unchecked.values()), retain_graph=True, allow_unused=True)
            for (name, _), gradient in zip(unchecked.items(), gradients):
                norm = float(gradient.norm()) if gradient is not None else 0.
                if not math.isfinite(norm) or norm <= 0:
                    raise RuntimeError('Attribute DATA objective has no nonzero '+name+' gradient')
                self.verified.add(name)
                print('[ATTRIBUTE DATA GRAD] {}={:.8g}'.format(name, norm), flush=True)
        if not self.probe_path_verified:
            source_gradients = torch.autograd.grad(data,
                (g._thermal_features_dc, g._thermal_features_rest, g._xyz,
                 g._scaling, g._rotation, g._opacity), retain_graph=True, allow_unused=True)
            if any(x is not None and bool((x != 0).any()) for x in source_gradients):
                raise RuntimeError('Attribute optimizer objective leaked into main Gaussian/SH parameters')
            self.probe_path_verified = True
            print('[ATTRIBUTE DATA PATH] main Gaussian/SH gradients=0; separate attribute optimizer', flush=True)
        (data+prior).backward()
        field.temperature_raw.grad[~field.supported] = 0
        for parameter in (field.epsilon_raw, field.slope_raw):
            if parameter.grad is not None:
                parameter.grad[~field.active_regions] = 0
        for parameter in field.parameters():
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise FloatingPointError('Nonfinite attribute gradient')
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.updates += 1
        return dict(DN_Huber=float(dn.detach()), metal_Huber=float(metal.detach()),
                    SH_display_Huber=float(sh.detach()), prior=float(prior.detach()))

    def loss(self, iteration, view, g, scene, geo, pipe, background, image, gt, deform):
        if iteration <= self.start:
            return image.new_zeros(())
        if self.problem is None:
            self._initialize(g)
        data_stats = self._update(iteration, view, g, scene, geo, pipe, background)
        from tools.probe_full_sh_attributes import TEMPERATURE_VARIANT
        p, field = self.problem, self.problem.field
        with torch.no_grad():
            values = field.values(TEMPERATURE_VARIANT, self.last_step)
            means = g.get_xyz if pipe.convert_SHs_python else g.get_xyz+deform[0]
            direction = F.normalize(means-view.camera_center[None], dim=1)
            basis = ((direction[field.metal_indices].double()-p.direction_mean)/p.direction_std).float()
            q = field.signal(values, basis, TEMPERATURE_VARIANT, self.last_step)
            static = (values['emissivity']*p.response.forward(values['temperature'])
                      +(1-values['emissivity'])*values['environment_signal'])
            lo = p.response.forward(q.new_tensor(p.options.temp_min))
            hi = p.response.forward(q.new_tensor(p.options.temp_max))
            span = (hi-lo).clamp_min(1e-8)
            adapter = SimpleNamespace(get_xyz=g.get_xyz,
                has_radiative_transfer_attributes=lambda:True,
                get_passband_radiance=((q-lo)/span)[:, None],
                get_radiometric_residual=((q-static)/span)[:, None],
                get_thermal_temperature=values['temperature'][:, None])
            colors = radiative_confidence_values(adapter, self.opt).expand(-1, 3).contiguous()
            confidence = render(view, g, pipe, background, *deform, g.get_xyz.device,
                is_6dof=self.dataset.is_6dof, feature_set='thermal', override_color=colors,
                detach_geometry=True, disable_atmosphere=True)['render'].clamp(0, 1)
        auxiliary, stats = radiative_confidence_mse_loss(image, gt, confidence, self.opt, iteration)
        if not self.sh_verified:
            gradients = torch.autograd.grad(auxiliary,
                (g._thermal_features_dc, g._thermal_features_rest, g._features_dc, g._features_rest),
                retain_graph=True, allow_unused=True)
            norms = [float(x.norm()) if x is not None else 0. for x in gradients]
            if not all(math.isfinite(x) and x > 0 for x in norms[:2]) or any(x != 0 for x in norms[2:]):
                raise RuntimeError('Feedback must reach both IR SH groups, without reaching RGB SH: '+str(norms))
            self.sh_verified = True
            print('[ATTRIBUTE SH FEEDBACK GRAD] IR_DC={:.8g} IR_rest={:.8g}; RGB_SH=0; target=GT'.format(*norms[:2]), flush=True)
        if iteration % 1000 == 0 or iteration == self.opt.iterations:
            row = dict(iteration=iteration, physical_step=self.last_step, updates=self.updates,
                target='observed_IR_GT', confidence='inherited_heuristic_not_calibrated_uncertainty',
                verified_data_gradients=sorted(self.verified), IR_SH_feedback_verified=self.sh_verified,
                slope_active=field.slope_active(TEMPERATURE_VARIANT, self.last_step),
                slope_min=float(values['cluster_slope_per_K'].min()),
                slope_max=float(values['cluster_slope_per_K'].max()), **data_stats, **stats)
            with self.log_path.open('a') as out:
                out.write(json.dumps(row, allow_nan=False)+'\n')
            print('[ATTRIBUTE FEEDBACK] '+json.dumps(row), flush=True)
        return auxiliary

    def save(self, iteration):
        if self.problem is None or iteration > self.opt.iterations:
            return  # Preserve the baseline best-PSNR mirror as SH-only.
        from tools.probe_full_sh_attributes import TEMPERATURE_VARIANT
        from utils.attribute_probe_pipeline import _sha, _json, _validate_emissivity_fields
        p, field = self.problem, self.problem.field
        with torch.no_grad():
            values = field.values(TEMPERATURE_VARIANT, self.last_step)
        folder = self.model/'point_cloud'/('iteration_%d'%iteration)
        geometry = {name:_sha(self.model/name/('iteration_%d'%iteration)/'geometry.pth')
                    for name in ('Geometry', 'GeometryRGB')}
        cache = dict(kind=LIVE_KIND, schema_version=2, variant=TEMPERATURE_VARIANT,
            source_model=str(self.model), source_iteration=iteration, step=self.last_step,
            physical_start=self.start, attribute_updates=self.updates, views_per_attribute_update=1,
            options=vars(p.options),
            state_dict={k:v.detach().cpu() for k,v in field.state_dict().items()},
            optimizer_state=self.optimizer.state_dict(),
            fields={k:v.detach().cpu() for k,v in values.items()},
            supported=field.supported.cpu(), material_ids=field.ids.cpu(), material_class=p.labels.cpu(),
            emissivity_model='linear_temperature', slope_learnable=True, slope_per_K=None,
            temperature_ref_K=p.options.temperature_ref, camera_response=p.response.c,
            temperature_ground_truth=False, emissivity_ground_truth=False,
            main_loss_feedback=True, feedback_target='observed_IR_GT',
            confidence_is_calibrated_uncertainty=False,
            verified_data_gradients=sorted(self.verified), IR_SH_feedback_verified=self.sh_verified,
            attribute_data_main_gradient_leak_check=self.probe_path_verified,
            main_integration=dict(kind='live_attribute_GT_MSE_feedback',
                point_cloud_sha256=_sha(folder/'point_cloud.ply'), geometry_sha256=geometry,
                fit_output=str(self.log_path.parent), main_loss_feedback=True))
        _validate_emissivity_fields(cache)
        target=folder/'radiometric_probe.pt'
        temporary=target.with_suffix('.pt.tmp');torch.save(cache,temporary);temporary.replace(target)
        pointer = self.model/'phys_probe'/('iteration_%d'%iteration)/'attribute_fit.json'
        pointer.parent.mkdir(parents=True, exist_ok=True)
        _json(pointer,
              dict(cache=str(target), fit_output=str(self.log_path.parent), variant=TEMPERATURE_VARIANT,
                   selected_step=self.last_step, attribute_updates=self.updates, slope_learnable=True,
                   slope_per_K=None, main_loss_feedback=True, feedback_target='observed_IR_GT',
                   confidence_is_calibrated_uncertainty=False))
        print('[ATTRIBUTE FEEDBACK] saved learned epsilon(T) and GT-MSE feedback state: '+str(target), flush=True)

    def close(self):
        if self.optimizer is not None:
            self.optimizer.zero_grad(set_to_none=True)
        self.problem = self.optimizer = None
