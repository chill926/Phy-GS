#!/usr/bin/env python3
"""Independent staged T/epsilon readout from frozen full-SH views plus raw DN.

No 3DGS training, main-code edits or free per-point radiance.
Material regions can share a bounded linear epsilon(T), including its slope.
Constant-emissivity variants remain available as controls.
The directional term is an empirical, material-shared reflection hypothesis;
the resulting fields are prior-conditioned estimates, never temperature GT.
"""
import argparse
import copy
import csv
from datetime import datetime
import json
from pathlib import Path
import time
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion
import torch
import torch.nn.functional as F

from utils.probe_dn_utils import (CameraResponse, ShuffledViews, evenly, frozen_matrix,
                          inverse_grid, read_config, sha, write_json)
from utils.emissivity_models import endpoint_emissivity_parameters

TEMPERATURE_VARIANT = 'learn_eps_temperature_directional'
VARIANTS = ('fixed_eps_static', 'learn_eps_static', 'learn_eps_directional', TEMPERATURE_VARIANT)
SOURCE_PARAMETERS = ('_xyz', '_features_dc', '_features_rest', '_thermal_features_dc',
                     '_thermal_features_rest', '_scaling', '_rotation', '_opacity')


def options(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('-m', '--model', required=True)
    p.add_argument('--iteration', type=int, default=30000)
    p.add_argument('--mapping', default='output/dn_display_mapping/RoadBlock_20261021_195046/monotone_colorize_then_resize/mapping.json')
    p.add_argument('--output', default='')
    p.add_argument('--variants', nargs='+', choices=VARIANTS, default=list(VARIANTS))
    p.add_argument('--steps', type=int, default=1024, help='Adam updates per variant, not 3DGS iterations')
    p.add_argument('--views-per-step', type=int, default=4)
    p.add_argument('--fit-views', type=int, default=0, help='0=all 49 predefined fit views')
    p.add_argument('--max-eval-views', type=int, default=0, help='Positive counts only for smoke testing')
    p.add_argument('--export-views', type=int, default=0, help='0=all test views')
    p.add_argument('--eval-every', type=int, default=128)
    p.add_argument('--log-every', type=int, default=32)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--temperature-lr', type=float, default=.001)
    p.add_argument('--emissivity-lr', type=float, default=.003)
    p.add_argument('--environment-lr', type=float, default=.0001)
    p.add_argument('--direction-lr', type=float, default=.003)
    p.add_argument('--direction-start-fraction', type=float, default=.25)
    p.add_argument('--emissivity-start-fraction', type=float, default=.5)
    p.add_argument('--temperature-ref', type=float, default=300., help='Reference temperature for epsilon0, K')
    p.add_argument('--slope-start-fraction', type=float, default=.75)
    p.add_argument('--slope-lr-scale', type=float, default=.25, help='Relative to emissivity learning rate')
    p.add_argument('--slope-prior-weight', type=float, default=.001)
    p.add_argument('--direction-amplitude-dn', type=float, default=256.)
    p.add_argument('--direction-prior-weight', type=float, default=.001)
    p.add_argument('--emissivity-prior-weight', type=float, default=.01)
    p.add_argument('--environment-prior-weight', type=float, default=.01)
    p.add_argument('--sh-weight', type=float, default=.1)
    p.add_argument('--sh-pixels', type=int, default=4096)
    p.add_argument('--metal-weight', type=float, default=.25, help='Additional metal-core DN Huber term; identical for all variants')
    p.add_argument('--huber-beta', type=float, default=.02, help='DN error normalized by fit-only contrast')
    p.add_argument('--min-accumulation', type=float, default=.98)
    p.add_argument('--metal-init', type=float, default=.3)
    p.add_argument('--metal-half-range', type=float, default=.1)
    p.add_argument('--temp-min', type=float, default=260.)
    p.add_argument('--temp-max', type=float, default=360.)
    p.add_argument('--env-temp-min', type=float, default=200.)
    p.add_argument('--env-temp-max', type=float, default=360.)
    p.add_argument('--env-quantile', type=float, default=.1)
    p.add_argument('--dry-run', action='store_true')
    o = p.parse_args(argv)
    if min(o.steps, o.views_per_step, o.eval_every, o.log_every, o.sh_pixels) < 1 or o.seed < 0:
        p.error('Steps, batch sizes and intervals must be positive; seed nonnegative')
    if min(o.fit_views, o.max_eval_views, o.export_views) < 0 or len(set(o.variants)) != len(o.variants):
        p.error('Invalid view counts or repeated variants')
    positive = (o.temperature_lr, o.emissivity_lr, o.environment_lr, o.direction_lr, o.slope_lr_scale,
                o.huber_beta, o.direction_amplitude_dn, o.metal_half_range)
    if any(not np.isfinite(x) or x <= 0 for x in positive):
        p.error('Learning rates, range, amplitude and beta must be positive and finite')
    nonnegative = (o.sh_weight, o.metal_weight, o.emissivity_prior_weight,
                   o.environment_prior_weight, o.direction_prior_weight, o.slope_prior_weight)
    if any(not np.isfinite(x) or x < 0 for x in nonnegative):
        p.error('Loss weights must be finite and nonnegative')
    if not (0 < o.temp_min < o.temp_max and 0 < o.env_temp_min < o.env_temp_max
            and 0 < o.min_accumulation <= 1 and 0 <= o.env_quantile <= 1
            and 0 <= o.direction_start_fraction < o.emissivity_start_fraction < 1):
        p.error('Invalid temperature bounds, visibility or release stages')
    if not (o.temp_min <= o.temperature_ref <= o.temp_max
            and o.emissivity_start_fraction <= o.slope_start_fraction < 1):
        p.error('Reference temperature must be in range; release slope after emissivity and before the final step')
    if not 0 < o.metal_init-o.metal_half_range < o.metal_init+o.metal_half_range < 1:
        p.error('Metal interval must lie strictly inside (0,1)')
    return o


class Attributes(torch.nn.Module):
    def __init__(self, initial_t, ids, low, high, initial_eps, supported, response, env_initial,
                 metal_indices, region_index, fit_basis, fit_weights, o):
        super().__init__()
        self.o, self.response, self.env_initial = o, response, float(env_initial)
        for name, value in dict(ids=ids, low=low, high=high, initial_eps=initial_eps,
                                supported=supported, metal_indices=metal_indices,
                                region_index=region_index, fit_basis=fit_basis,
                                fit_weights=fit_weights).items():
            self.register_buffer(name, value)
        self.temperature_raw = torch.nn.Parameter(torch.logit(
            ((initial_t-o.temp_min)/(o.temp_max-o.temp_min)).clamp(1e-5, 1-1e-5)))
        self.epsilon_raw = torch.nn.Parameter(torch.logit((initial_eps-low)/(high-low)))
        # Zero difference gives exactly the old constant-emissivity initialization.
        self.slope_raw = torch.nn.Parameter(torch.zeros_like(initial_eps))
        self.environment_raw = torch.nn.Parameter(initial_t.new_tensor(float(np.log(
            (env_initial-o.env_temp_min)/(o.env_temp_max-env_initial)))))
        regions = int(region_index.max())+1 if len(region_index) else 0
        self.direction_raw = torch.nn.Parameter(torch.zeros(regions, 3, device=initial_t.device))
        active = torch.bincount(ids[supported], minlength=len(low)) > 0
        self.register_buffer('active_regions', active)

    def stages(self, variant, step):
        return (variant != 'fixed_eps_static' and step > int(self.o.steps*self.o.emissivity_start_fraction),
                variant in ('learn_eps_directional', TEMPERATURE_VARIANT)
                and step > int(self.o.steps*self.o.direction_start_fraction))

    def slope_active(self, variant, step):
        return variant == TEMPERATURE_VARIANT and step > int(self.o.steps*self.o.slope_start_fraction)

    def values(self, variant, step):
        learn_e, _ = self.stages(variant, step)
        raw_e = self.epsilon_raw if learn_e else self.epsilon_raw.detach()
        if variant == TEMPERATURE_VARIANT:
            settings = dict(emissivity_min=self.low, emissivity_max=self.high,
                            temperature_min_K=self.o.temp_min, temperature_max_K=self.o.temp_max,
                            temperature_ref_K=self.o.temperature_ref)
            eps, slope, e_low, e_high = endpoint_emissivity_parameters(
                raw_e, self.slope_raw, settings, freeze_slope=not self.slope_active(variant, step))
        else:
            eps = self.low+(self.high-self.low)*raw_e.sigmoid()
            slope, e_low, e_high = torch.zeros_like(eps), eps, eps
        t = self.o.temp_min+(self.o.temp_max-self.o.temp_min)*self.temperature_raw.sigmoid()
        e = eps[self.ids]+slope[self.ids]*(t-self.o.temperature_ref)
        env_t = self.o.env_temp_min+(self.o.env_temp_max-self.o.env_temp_min)*self.environment_raw.sigmoid()
        return dict(temperature=t, emissivity=e, cluster_emissivity=eps,
                    emissivity_ref=eps[self.ids], emissivity_slope_per_K=slope[self.ids],
                    cluster_slope_per_K=slope, cluster_emissivity_low=e_low, cluster_emissivity_high=e_high,
                    environment_K=env_t, environment_signal=self.response.forward(env_t))

    def direction(self, basis, variant, step):
        _, active = self.stages(variant, step)
        if not active or not len(self.metal_indices):
            return basis.new_zeros(len(self.metal_indices))
        coefficients = self.direction_raw[self.region_index]
        current = (coefficients*basis).sum(-1).tanh()
        # Remove the exact fit-view weighted mean AFTER tanh; a directional
        # deviation cannot freely change a point's fit-view mean emission.
        fit = (self.fit_basis*coefficients[None]).sum(-1).tanh()
        fit_mean = (fit*self.fit_weights).sum(0)/self.fit_weights.sum(0).clamp_min(1e-12)
        return .5*(current-fit_mean)  # Bounded [-1,1], fit-weighted mean zero.

    def signal(self, values, basis, variant, step):
        eps, env = values['emissivity'], values['environment_signal']
        signal = eps*self.response.forward(values['temperature'])+(1-eps)*env
        if len(self.metal_indices):
            amplitude = torch.minimum(env*.9, env.new_tensor(self.o.direction_amplitude_dn))
            adjustment = (1-eps[self.metal_indices])*amplitude*self.direction(basis, variant, step)
            signal = signal.index_add(0, self.metal_indices, adjustment)
        return signal

    def prior(self, values, variant, step):
        active = self.active_regions
        e = (values['cluster_emissivity'][active]-self.initial_eps[active])/(self.high-self.low)[active]
        loss = self.o.emissivity_prior_weight*F.smooth_l1_loss(e, torch.zeros_like(e), beta=.1)
        loss = loss+self.o.environment_prior_weight*((values['environment_K']-self.env_initial)/50.)**2
        if self.slope_active(variant, step):
            change = values['cluster_slope_per_K'][active]*(self.o.temp_max-self.o.temp_min)
            loss = loss+self.o.slope_prior_weight*(change/(self.high-self.low)[active]).square().mean()
        if self.stages(variant, step)[1] and self.direction_raw.numel():
            loss = loss+self.o.direction_prior_weight*self.direction_raw.square().mean()
        return loss


def display_pixels(mapping, width, height, native_valid, count, rng, device):
    """Sample a fixed, no-target-leakage pixel set; all four raw neighbors valid."""
    total = width*height
    idx = np.sort(rng.choice(total, min(count, total), replace=False))
    x, y = idx % width, idx // width
    xy = torch.tensor(np.stack([2*(x+.5)/width-1, 2*(y+.5)/height-1], -1)[None], device=device, dtype=torch.float32)
    p = mapping.geometry
    gx = (1+p[0]/width)*xy[..., 0]+p[1]/width*xy[..., 1]+2*p[2]/width
    gy = p[3]/height*xy[..., 0]+(1+p[4]/height)*xy[..., 1]+2*p[5]/height
    h, w = native_valid.shape
    sx, sy = (gx[0]+1)*w/2-.5, (gy[0]+1)*h/2-.5
    x0, y0 = sx.floor().long(), sy.floor().long()
    inside = (x0 >= 0) & (y0 >= 0) & (x0+1 < w) & (y0+1 < h)
    x0, y0 = x0.clamp(0, w-2), y0.clamp(0, h-2)
    valid = inside & native_valid[y0, x0] & native_valid[y0, x0+1] & native_valid[y0+1, x0] & native_valid[y0+1, x0+1]
    return xy, torch.tensor(idx, device=device), valid


def quantiles(x):
    return torch.quantile(x.float(), x.new_tensor([.05, .5, .95])).cpu().tolist() if x.numel() else None


def main(argv=None, *, prepare_only=False):
    o = options(argv)
    model, mapping_path = (ROOT/o.model).resolve(), (ROOT/o.mapping).resolve()
    cfg = read_config(model, torch.device('cuda:0'))
    scene_path = Path(cfg.source_path)
    document = json.loads(mapping_path.read_text())
    if (ROOT/document['settings']['scene']).resolve() != scene_path or document.get('test_used_for_fitting_or_selection') is not False:
        raise ValueError('Frozen mapping scene/split provenance mismatch')
    if document['order'] != 'colorize_then_resize':
        raise ValueError('This experiment requires the previously validated colorize-then-resize mapping')
    cache_path = model/'phys_probe'/('iteration_%d'%o.iteration)/'full_sh_observations.pt'
    cache = torch.load(cache_path, map_location='cpu')
    if (cache.get('kind') != 'full_sh_probe_input_observations'
            or int(cache.get('source_iteration', -1)) != o.iteration
            or Path(cache.get('source_model', '')).resolve() != model):
        raise ValueError('Require matching new full-SH observations, not a legacy physical cache')
    # Only SH coefficients and fixed SAM3 labels are reused. In particular the
    # cached 62-view means/weights are NOT read as initialization (13 val views).
    assets = [model/'cfg_args', model/'point_cloud'/('iteration_%d'%o.iteration)/'point_cloud.ply',
              model/'Geometry'/('iteration_%d'%o.iteration)/'geometry.pth',
              model/'GeometryRGB'/('iteration_%d'%o.iteration)/'geometry.pth', mapping_path, cache_path,
              ROOT/'utils/probe_dn_utils.py', ROOT/'tools/fit_flir_display_mapping.py']
    for folder in ('scene', 'utils', 'arguments', 'gaussian_renderer'):
        assets.extend(sorted((ROOT/folder).rglob('*.py')))
    assets.extend([ROOT/'train.py', ROOT/'render.py'])
    before = {str(p): sha(p) for p in assets}
    from tools.fit_flir_display_mapping import extract_flir, load_model
    roles = dict(fit=evenly(sorted(document['fit_frames']), o.fit_views),
                 validation=evenly(sorted(document['validation_frames']), o.max_eval_views),
                 test=evenly(sorted(p.stem for p in (scene_path/'thermal/test').glob('*.jpg')), o.max_eval_views))
    if not all(roles.values()) or len(set(sum(roles.values(), []))) != sum(map(len, roles.values())):
        raise ValueError('Require disjoint nonempty fit, validation and test views')
    targets, response = {}, None
    for role, names in roles.items():
        for name in names:
            raw = scene_path/'raw_images'/(name+'.jpg')
            dn, palette, meta = extract_flir(raw, document['png_byte_order'])
            if response is None:
                response = CameraResponse(meta['planck'])
            if meta['planck'] != response.c or not np.array_equal(palette, np.asarray(document['palette_ycrcb'], dtype=np.uint8)):
                raise ValueError('Camera response/palette mismatch: '+name)
            for branch in ('thermal', 'rgb'):
                path = scene_path/branch/('test' if role == 'test' else 'train')/(name+'.jpg')
                before[str(path)] = sha(path)
                if Image.open(path).size != (640, 480):
                    raise ValueError('Frozen mapping requires 640x480 images')
            before[str(raw)] = sha(raw)
            targets[name] = dict(dn=dn, signal=dn.astype(np.float32)+response.c['O'], role=role, meta=meta)
    shapes = {t['dn'].shape for t in targets.values()}
    if len(shapes) != 1:
        raise ValueError('Mixed raw grid shapes')
    shape = next(iter(shapes))
    matrix = frozen_matrix(document, (shape[1], shape[0]))
    metadata = dict(kind='independent_full_sh_radiometric_attribute_probe', revision=2,
                    script_sha256=sha(__file__), source_model=str(model), source_iteration=o.iteration,
                    split_camera_names=roles, camera_response=response.c,
                    frozen_mapping=str(mapping_path), mapping_fitted=False,
                    source_model_optimized=False, main_loss_feedback=False,
                    temperature_ground_truth=False, emissivity_ground_truth=False,
                    full_SH_used='direction-evaluated full RGB render; display-space auxiliary loss',
                    raw_DN_used='primary physical-unit image-space constraint',
                    per_point_free_radiance=False,
                    temperature_emissivity_variant=TEMPERATURE_VARIANT,
                    temperature_ref_K=o.temperature_ref,
                    slope_parameterization='material_region_shared_bounded_endpoint_logits; zero initialization',
                    cached_62_view_aggregate_used_for_initialization=False,
                    geometry_and_material_prior_already_saw_published_train_including_probe_validation=True,
                    release_steps=dict(direction=int(o.steps*o.direction_start_fraction)+1,
                                       emissivity=int(o.steps*o.emissivity_start_fraction)+1,
                                       slope=int(o.steps*o.slope_start_fraction)+1),
                    selection='lowest validation balanced DN Huber; no test selection',
                    direction_hypothesis='bounded first-order view-direction deviation shared within supported metal depth/material regions, exactly zero fit-weighted mean per point; not a measured BRDF',
                    target_to_native_matrix=matrix.tolist(), options=vars(o))
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)
    if o.dry_run:
        print('DRY RUN PASSED; no output/checkpoint written.', flush=True)
        return
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA renderer required')
    device = torch.device('cuda:0'); torch.cuda.set_device(device); torch.manual_seed(0)
    output = (ROOT/o.output).resolve() if o.output else ROOT/'output'/('probe_full_sh_attributes_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    if output.exists():
        raise FileExistsError('Choose a new output directory')
    for protected in (model, scene_path):
        if output == protected or protected in output.parents or output in protected.parents:
            raise ValueError('Output must be separate from source model and dataset')
    output.mkdir(parents=True)
    write_json(output/'provenance.json', metadata)
    write_json(output/'options.json', vars(o))
    from scene import Scene, GaussianModel, GeoRefineModel
    from arguments import PipelineParams
    from gaussian_renderer import render
    from train import compute_geo_fields, estimate_temporal_jitter_interval
    from utils.sh_utils import eval_sh
    g = GaussianModel(cfg.sh_degree, device)
    scene = Scene(cfg, g, load_iteration=o.iteration, shuffle=False)
    for name in SOURCE_PARAMETERS:
        getattr(g, name).requires_grad_(False)
    n = len(g.get_xyz)
    if not torch.equal(g.get_thermal_features.detach().cpu(), cache['thermal_sh_coefficients']) or not torch.equal(g.get_xyz.detach().cpu(), cache['base_xyz']):
        raise ValueError('Full-SH cache does not match the actual point order/weights')
    ids = cache['material_ids'].reshape(-1).to(device).long()
    classes = cache['cluster_class'].reshape(-1).to(device).long()
    labels = cache['material_class'].reshape(-1).to(device).long()
    if len(ids) != n or ids.min() < 0 or ids.max() >= len(classes) or not torch.equal(classes[ids], labels):
        raise ValueError('Invalid material partition cache')
    from utils.material_config import load_material_config, material_bounds
    material_config, config_path, config_hash = load_material_config(cfg.phys_material_config)
    before[str(config_path)] = sha(config_path)
    if cache['material_partition']['config_sha256'] != config_hash:
        raise ValueError('Material config changed since cache generation')
    cfg.phys_metal_e_init, cfg.phys_metal_e_half_range = o.metal_init, o.metal_half_range
    low, high, initial_eps = material_bounds(material_config, cfg, classes, device)
    pipe = PipelineParams(argparse.ArgumentParser(add_help=False))
    for key in vars(pipe):
        if hasattr(cfg, key):
            setattr(pipe, key, getattr(cfg, key))
    kwargs = dict(time_multires=cfg.geometry_time_multires) if cfg.geometry_time_multires >= 0 else {}
    geo = GeoRefineModel(device, cfg.is_blender, False, enabled=cfg.ir_geometry_mode != 'static', **kwargs)
    rgb_geo = GeoRefineModel(device, cfg.is_blender, False, **kwargs)
    geo.load_weights(str(model), iteration=o.iteration)
    rgb_geo.load_weights(str(model), iteration=o.iteration, subdir='GeometryRGB')
    for network in (geo, rgb_geo):
        network.geometry.eval()
        for param in network.geometry.parameters():
            param.requires_grad_(False)
    display, loaded_document = load_model(mapping_path, device)
    for param in display.parameters():
        param.requires_grad_(False)
    black = torch.zeros(3, device=device)
    background = torch.ones_like(black) if cfg.white_background else black
    interval = estimate_temporal_jitter_interval(scene.getTrainCameras(), 'fid_median')
    cameras = {v.image_name: v for v in scene.getTrainCameras()+scene.getTestCameras()}
    grid, fov = inverse_grid(matrix, shape, device)
    rng = np.random.default_rng(o.seed)
    frames, ref_checks = {}, []

    def deformation(view, network=geo, feature_set='thermal'):
        return tuple(x.detach() if torch.is_tensor(x) else x for x in compute_geo_fields(
            view, g, network, pipe, background, cfg, o.iteration, lambda _: 0., interval,
            device, cfg.is_blender, feature_set=feature_set, eval_mode=True))

    def raster(frame, colors=None, bg=None):
        return render(frame['view'], g, pipe, background if bg is None else bg, *frame['deformation'], device,
                      feature_set='thermal', override_color=colors, detach_geometry=True, disable_atmosphere=True)['render']

    def native(image):
        return F.grid_sample(image[None], grid, align_corners=False, padding_mode='zeros')[0]

    print('[PREPARE] frozen IR geometry, full-SH RGB and native DN', flush=True)
    with torch.no_grad():
        for name, target in targets.items():
            view = cameras[name]; view.load2device(device)
            frame = dict(view=view, deformation=deformation(view), role=target['role'], meta=target['meta'])
            acc = native(raster(frame, torch.ones(n, 3, device=device), black))[0]
            valid = fov & torch.isfinite(acc) & (acc >= o.min_accumulation)
            if valid.sum() < 100:
                raise ValueError('Too few opaque pixels: '+name)
            full_sh = raster(frame).clamp(0, 1)
            means = g.get_xyz if pipe.convert_SHs_python else g.get_xyz+frame['deformation'][0]
            direction = F.normalize(means-view.camera_center[None], dim=1)
            rgb = (eval_sh(g.active_sh_degree, g.get_thermal_features.transpose(1, 2), direction)+.5).clamp_min(0)
            match = float((raster(frame, rgb.contiguous()).clamp(0, 1)-full_sh).abs().max())
            if not np.isfinite(match) or match > 2e-5:
                raise RuntimeError('Full-SH renderer reference mismatch: '+name)
            ref_checks.append(dict(view=name, max_error=match))
            metal_probability = native(raster(frame, (labels == 2).float()[:, None].expand(-1, 3).contiguous(), black))[0]/acc.clamp_min(1e-8)
            core = torch.tensor(binary_erosion(((metal_probability > .9) & valid).cpu().numpy(), iterations=1), device=device)
            xy, sample_indices, display_valid = display_pixels(display, 640, 480, valid, o.sh_pixels, rng, device)
            teacher = full_sh.flatten(1).T[sample_indices]
            informative = (teacher.amax(-1) > .01) & (teacher.amin(-1) < .99)
            display_valid &= informative
            target_signal = torch.tensor(target['signal'], device=device)
            frame.update(acc=acc, valid=valid, metal_core=core, target=target_signal,
                         directions=direction, xy=xy, sample_indices=sample_indices,
                         display_valid=display_valid, teacher=teacher)
            frames[name] = frame
            print('[PREPARE] {} {} opaque={} metal={} display={}'.format(
                target['role'], name, int(valid.sum()), int(core.sum()), int(display_valid.sum())), flush=True)
    observed = torch.cat([frames[name]['target'][frames[name]['valid']] for name in roles['fit']])
    loss_scale = float((observed.quantile(.95)-observed.quantile(.05)).clamp_min(1))
    signal_scale = float(observed.median())
    env_signal = observed.quantile(o.env_quantile)
    env_initial = float(response.inverse(env_signal))
    if not o.env_temp_min < env_initial < o.env_temp_max:
        raise ValueError('Environment initialization outside specified bounds')
    # Fit-only adjoint initialization. Include a radiometric environment
    # background (1-acc)*env; do NOT divide a DN radiance render by acc.
    weights, numerator = [], torch.zeros(n, device=device)
    for name in roles['fit']:
        frame = frames[name]
        color = torch.zeros(n, 3, device=device, requires_grad=True)
        image = native(raster(frame, color, black))
        corrected = (frame['target']-(1-frame['acc'])*env_signal)/frame['acc'].clamp_min(1e-8)
        objective = ((image[0]*(corrected/signal_scale)+image[1])*frame['valid']).sum()
        grad = torch.autograd.grad(objective, color)[0].detach()
        numerator += grad[:, 0]
        weights.append(grad[:, 1].clamp_min(0))
    fit_weights = torch.stack(weights)
    total_weight = fit_weights.sum(0)
    supported = total_weight > 1e-7
    if not supported.any():
        raise RuntimeError('No fit-supported Gaussians')
    mean_signal = torch.where(supported, numerator/total_weight.clamp_min(1e-8)*signal_scale, observed.median())
    initial_bb = (mean_signal-(1-initial_eps[ids])*env_signal)/initial_eps[ids]
    bb_low = response.forward(observed.new_tensor(o.temp_min))
    bb_high = response.forward(observed.new_tensor(o.temp_max))
    initial_t = response.inverse(initial_bb.clamp(bb_low, bb_high)).clamp(o.temp_min+.01, o.temp_max-.01)
    effective = total_weight.double().square()/fit_weights.double().square().sum(0).clamp_min(1e-30)
    metal_indices = torch.where(supported & (labels == 2) & (effective >= 3))[0]
    if len(metal_indices):
        _, region_index = torch.unique(ids[metal_indices], sorted=True, return_inverse=True)
        direction_fit = torch.stack([frames[name]['directions'][metal_indices] for name in roles['fit']])
        w_metal = fit_weights[:, metal_indices]
        direction_mean = (direction_fit.double()*w_metal.double()[..., None]).sum(0)/w_metal.double().sum(0)[:, None].clamp_min(1e-30)
        direction_std = ((direction_fit.double()-direction_mean).square()*w_metal.double()[..., None]).sum(0)/w_metal.double().sum(0)[:, None].clamp_min(1e-30)
        direction_std = direction_std.sqrt().clamp_min(.05)
        fit_basis = ((direction_fit.double()-direction_mean)/direction_std).float()
        normalized_weights = w_metal/w_metal.sum(0).clamp_min(1e-12)
    else:
        region_index = torch.empty(0, device=device, dtype=torch.long)
        direction_mean = direction_std = torch.empty(0, 3, device=device)
        fit_basis = torch.empty(len(roles['fit']), 0, 3, device=device)
        normalized_weights = torch.empty(len(roles['fit']), 0, device=device)
    for frame in frames.values():
        frame['basis'] = ((frame['directions'][metal_indices].double()-direction_mean)/direction_std).float()
        with torch.no_grad():
            support_fraction = native(raster(frame, supported.float()[:, None].expand(-1, 3).contiguous(), black))[0]/frame['acc'].clamp_min(1e-8)
            frame['supported_valid'] = frame['valid'] & (support_fraction >= .95)
    initialization = dict(points=n, fit_supported_points=int(supported.sum()), directional_metal_points=len(metal_indices),
                          directional_regions=int(region_index.max())+1 if len(region_index) else 0,
                          loss_scale=loss_scale, signal_scale=signal_scale, environment_initial_K=env_initial,
                          initial_temperature_clipped_fraction=float(((initial_bb < bb_low) | (initial_bb > bb_high))[supported].float().mean()),
                          temperature_initialization='fit-only alpha-adjoint DN observation backprojection; not pointwise ground truth',
                          prediction='sum(alpha_transmittance*physical_signal)+(1-acc)*environment_signal, no acc division',
                          field_visualization='visibility-normalized field averages, only well-supported opaque pixels')
    write_json(output/'initialization.json', initialization)
    write_json(output/'reference_checks.json', ref_checks)
    print('[INITIALIZATION] '+json.dumps(initialization), flush=True)

    def create_field():
        return Attributes(initial_t, ids, low, high, initial_eps, supported, response, env_initial,
                          metal_indices, region_index, fit_basis, normalized_weights, o).to(device)

    if prepare_only:
        from types import SimpleNamespace
        # Return only detached observations and field variables. The temporary
        # Gaussian/geometry models and their render closures are not retained.
        keep = ('target', 'meta', 'xy', 'sample_indices', 'display_valid', 'teacher')
        fit_frames = {name:{key:frames[name][key] for key in keep} for name in roles['fit']}
        problem = SimpleNamespace(field=create_field(), options=o, frames=fit_frames,
            roles=roles, grid=grid.detach(), fov=fov.detach(), labels=labels.detach(),
            display=display, response=response, signal_scale=signal_scale,
            loss_scale=loss_scale, direction_mean=direction_mean.detach(),
            direction_std=direction_std.detach(), mapping_path=str(mapping_path),
            source_model=str(model), source_iteration=o.iteration,
            preparation_output=str(output))
        print('[ATTRIBUTE FEEDBACK] prepared fit-only observations; no independent fit', flush=True)
        return problem

    initial_state = copy.deepcopy(create_field().state_dict())

    def predict(field, frame, variant, step, values=None):
        values = field.values(variant, step) if values is None else values
        signal = field.signal(values, frame['basis'], variant, step)
        color = (signal/signal_scale)[:, None].expand(-1, 3).contiguous()
        q = native(raster(frame, color, black))[0]*signal_scale+(1-frame['acc'])*values['environment_signal']
        return q

    def display_prediction(q, frame):
        dn = q-response.c['O']
        normalized = (dn-frame['meta']['median'])/frame['meta']['span']+.5
        return display(normalized[None], frame['xy'], (480, 640))[0]

    def data_terms(field, frame, variant, step):
        values = field.values(variant, step)
        prediction = predict(field, frame, variant, step, values)
        error = (prediction-frame['target'])/loss_scale
        loss_all = F.smooth_l1_loss(error[frame['valid']], torch.zeros_like(error[frame['valid']]), beta=o.huber_beta)
        core = frame['metal_core']
        loss_metal = F.smooth_l1_loss(error[core], torch.zeros_like(error[core]), beta=o.huber_beta) if core.sum() >= 4 else prediction.new_tensor(0.)
        display_valid = frame['display_valid']
        predicted_rgb = display_prediction(prediction, frame)
        loss_rgb = F.smooth_l1_loss(predicted_rgb[display_valid], frame['teacher'][display_valid], beta=.02) if display_valid.any() else prediction.new_tensor(0.)
        return loss_all+o.metal_weight*loss_metal+o.sh_weight*loss_rgb, (loss_all, loss_metal, loss_rgb), prediction

    @torch.no_grad()
    def evaluate(field, variant, step, role):
        rows = []
        for name in roles[role]:
            frame = frames[name]
            total, components, prediction = data_terms(field, frame, variant, step)
            error = prediction-frame['target']
            core = frame['metal_core']
            support = frame['supported_valid']
            rows.append(dict(view=name, MAE_DN=float(error[frame['valid']].abs().mean()),
                             MSE_DN=float(error[frame['valid']].square().mean()),
                             supported_pixels=int(support.sum()), supported_SSE=float(error[support].square().sum()),
                             metal_pixels=int(core.sum()), metal_SSE=float(error[core].square().sum()),
                             metal_SAE=float(error[core].abs().sum()),
                             DN_Huber=float(components[0]), metal_Huber=float(components[1]),
                             SH_display_Huber=float(components[2]), objective=float(total)))
        metal_count = sum(r['metal_pixels'] for r in rows)
        supported_count = sum(r['supported_pixels'] for r in rows)
        mean = lambda key: float(np.mean([r[key] for r in rows]))
        return dict(MAE_DN=mean('MAE_DN'), RMSE_DN=float(np.sqrt(mean('MSE_DN'))),
                    fit_supported_RMSE_DN=float(np.sqrt(sum(r['supported_SSE'] for r in rows)/supported_count)) if supported_count else None,
                    fit_supported_pixels=supported_count,
                    metal_core_RMSE_DN=float(np.sqrt(sum(r['metal_SSE'] for r in rows)/metal_count)) if metal_count else None,
                    metal_core_MAE_DN=sum(r['metal_SAE'] for r in rows)/metal_count if metal_count else None,
                    metal_pixels=metal_count, balanced_DN_Huber=mean('DN_Huber')+o.metal_weight*mean('metal_Huber'),
                    SH_display_Huber=mean('SH_display_Huber'), objective=mean('objective'), per_view=rows)

    def checkpoint(path, field, variant, step, metrics):
        values = field.values(variant, step)
        torch.save(dict(kind=metadata['kind'], schema_version=2, variant=variant, step=step,
                        source_model=str(model), source_iteration=o.iteration, options=vars(o),
                        state_dict={k:v.detach().cpu() for k,v in field.state_dict().items()},
                        fields={k:v.detach().cpu() for k,v in values.items()},
                        temperature_ground_truth=False, emissivity_ground_truth=False,
                        emissivity_model='linear_temperature' if variant == TEMPERATURE_VARIANT else 'constant',
                        slope_learnable=variant == TEMPERATURE_VARIANT,
                        slope_per_K=None if variant == TEMPERATURE_VARIANT else 0.,
                        temperature_ref_K=o.temperature_ref,
                        supported=supported.cpu(), material_ids=ids.cpu(), material_class=labels.cpu(),
                        camera_response=response.c, target_to_native=matrix,
                        directional_metal_indices=metal_indices.cpu(), direction_mean=direction_mean.cpu(),
                        direction_std=direction_std.cpu(), fit_view_names=roles['fit'], validation_metrics=metrics), path)

    results = []
    for variant in o.variants:
        folder = output/variant; folder.mkdir()
        field = create_field(); field.load_state_dict(initial_state)
        optimizer = torch.optim.Adam([dict(params=[field.temperature_raw], lr=o.temperature_lr),
                                      dict(params=[field.epsilon_raw], lr=o.emissivity_lr),
                                      dict(params=[field.slope_raw], lr=o.emissivity_lr*o.slope_lr_scale),
                                      dict(params=[field.environment_raw], lr=o.environment_lr),
                                      dict(params=[field.direction_raw], lr=o.direction_lr)])
        sampler = ShuffledViews(roles['fit'], o.seed)
        history = []

        def record(step):
            metrics = {role:evaluate(field, variant, step, role) for role in ('fit', 'validation')}
            row = dict(step=step, fit_views_seen=sampler.views_seen, fit_passes=sampler.views_seen/len(roles['fit']))
            for role, values in metrics.items():
                row[role] = {k:v for k,v in values.items() if k != 'per_view'}
                with (folder/(role+'.jsonl')).open('a') as file:
                    file.write(json.dumps(dict(step=step, **values), allow_nan=False)+'\n')
            history.append(row)
            with (folder/'metrics.jsonl').open('a') as file:
                file.write(json.dumps(row, allow_nan=False)+'\n')
            print('[{}][EVAL] {}/{} fit RMSE={:.3f} val={:.3f} metal={} DN | score={:.6g}'.format(
                variant, step, o.steps, metrics['fit']['RMSE_DN'], metrics['validation']['RMSE_DN'],
                metrics['validation']['metal_core_RMSE_DN'], metrics['validation']['balanced_DN_Huber']), flush=True)
            return metrics

        start = time.monotonic()
        initial_metrics = record(0)
        best, best_step = copy.deepcopy(initial_metrics['validation']), 0
        checkpoint(folder/'probe_best.pt', field, variant, 0, best)
        for step in range(1, o.steps+1):
            names = sampler.take(o.views_per_step)
            optimizer.zero_grad(set_to_none=True)
            components = np.zeros(3)
            for name in names:
                loss, parts, _ = data_terms(field, frames[name], variant, step)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite attribute data loss')
                (loss/len(names)).backward()
                components += np.asarray([float(v.detach()) for v in parts])/len(names)
            prior = field.prior(field.values(variant, step), variant, step)
            prior.backward()
            if field.temperature_raw.grad is None:
                raise RuntimeError('Missing temperature gradient')
            field.temperature_raw.grad[~supported] = 0
            if field.epsilon_raw.grad is not None:
                field.epsilon_raw.grad[~field.active_regions] = 0
            if field.slope_raw.grad is not None:
                field.slope_raw.grad[~field.active_regions] = 0
            for parameter in field.parameters():
                if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError('Nonfinite attribute gradient')
            gradient = dict(temperature=float(field.temperature_raw.grad.norm()),
                            emissivity=float(field.epsilon_raw.grad.norm()) if field.epsilon_raw.grad is not None else 0.,
                            slope=float(field.slope_raw.grad.norm()) if field.slope_raw.grad is not None else 0.,
                            direction=float(field.direction_raw.grad.norm()) if field.direction_raw.grad is not None else 0.)
            optimizer.step()
            row = dict(step=step, views=names, DN_Huber=components[0], metal_Huber=components[1],
                       SH_display_Huber=components[2], prior=float(prior.detach()), gradients=gradient,
                       release=dict(zip(('emissivity', 'direction'), field.stages(variant, step)),
                                    slope=field.slope_active(variant, step)))
            with (folder/'fit_log.jsonl').open('a') as file:
                file.write(json.dumps(row, allow_nan=False)+'\n')
            if step == 1 or step % o.log_every == 0 or step == o.steps:
                print('[{}] {}/{} DN={:.6g} metal={:.6g} SH={:.6g} | grad {}'.format(
                    variant, step, o.steps, *components, gradient), flush=True)
                if variant == TEMPERATURE_VARIANT:
                    with torch.no_grad():
                        slopes = field.values(variant, step)['cluster_slope_per_K'][field.active_regions]
                        print('[EPSILON(T)] slope_active={} k/K min={:.6g} mean={:.6g} max={:.6g}'.format(
                            field.slope_active(variant, step), float(slopes.min()),
                            float(slopes.mean()), float(slopes.max())), flush=True)
            if step % o.eval_every == 0 or step == o.steps:
                current = record(step)
                if current['validation']['balanced_DN_Huber'] < best['balanced_DN_Huber']:
                    best, best_step = copy.deepcopy(current['validation']), step
                    checkpoint(folder/'probe_best.pt', field, variant, step, best)
        checkpoint(folder/'probe_final.pt', field, variant, o.steps, current['validation'])
        best_state = torch.load(folder/'probe_best.pt', map_location=device)
        field.load_state_dict(best_state['state_dict'])
        with torch.no_grad():
            values = field.values(variant, best_step)
            endpoints = torch.stack([values['cluster_emissivity_low'], values['cluster_emissivity_high']])
            if (not ((values['emissivity'] >= low[ids]-1e-6) & (values['emissivity'] <= high[ids]+1e-6)).all()
                    or not ((endpoints >= low-1e-6) & (endpoints <= high+1e-6)).all()):
                raise RuntimeError('Emissivity bounds violated')
            metal = supported & (labels == 2)
            fit = evaluate(field, variant, best_step, 'fit')
            test = evaluate(field, variant, best_step, 'test')
            summary = dict(variant=variant, best_step=best_step, steps=o.steps,
                           elapsed_seconds=time.monotonic()-start, initial=initial_metrics,
                           final=current, fit=fit, validation=best, test=test,
                           metal_temperature_K_p05_p50_p95=quantiles(values['temperature'][metal]),
                           metal_emissivity_p05_p50_p95=quantiles(values['emissivity'][metal]),
                           environment_K=float(values['environment_K']),
                           emissivity_model='linear_temperature' if variant == TEMPERATURE_VARIANT else 'constant',
                           slope_learnable=variant == TEMPERATURE_VARIANT,
                           slope_per_K=None if variant == TEMPERATURE_VARIANT else 0.,
                           temperature_ref_K=o.temperature_ref,
                           cluster_slope_per_K_min_mean_max=[float(fn(values['cluster_slope_per_K'][field.active_regions]))
                                                           for fn in (torch.min, torch.mean, torch.max)],
                           metal_slope_per_K_p05_p50_p95=quantiles(values['emissivity_slope_per_K'][metal]),
                           direction_coefficients=field.direction_raw.cpu().tolist(),
                           temperature_near_low_fraction=float((values['temperature'][supported] < o.temp_min+.5).float().mean()),
                           temperature_near_high_fraction=float((values['temperature'][supported] > o.temp_max-.5).float().mean()))
        write_json(folder/'summary.json', summary)
        export_previews(folder, field, variant, best_step, roles, frames, o, g, pipe, rgb_geo,
                        cfg, background, response, supported, deformation, raster, native, predict, display)
        plot_history(folder, history)
        results.append(summary)
        print('[DONE] {} selected={} val={:.3f} test={:.3f} metal={} DN'.format(
            variant, best_step, best['RMSE_DN'], test['RMSE_DN'], test['metal_core_RMSE_DN']), flush=True)

    after = {p:sha(p) for p in before}
    if after != before:
        raise RuntimeError('A source checkpoint, input, mapping or main source file changed')
    for name in SOURCE_PARAMETERS:
        if getattr(g, name).grad is not None:
            raise RuntimeError('Source model received gradients: '+name)
    for module in (geo.geometry, rgb_geo.geometry, display):
        if any(p.grad is not None for p in module.parameters()):
            raise RuntimeError('Frozen geometry/mapping received gradients')
    write_json(output/'summary.json', dict(metadata, initialization=initialization, variants=results,
                                          source_assets_unchanged=True, source_sha256_before=before,
                                          source_sha256_after=after,
                                          best_variant_by_validation=min(results, key=lambda r:r['validation']['balanced_DN_Huber'])['variant']))
    with (output/'summary.csv').open('w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['variant', 'best_step', 'fit_RMSE_DN', 'val_RMSE_DN', 'test_RMSE_DN',
                         'val_metal_RMSE_DN', 'test_metal_RMSE_DN', 'metal_T_median_C', 'metal_epsilon_median',
                         'cluster_slope_min_per_K', 'cluster_slope_mean_per_K', 'cluster_slope_max_per_K'])
        for r in results:
            t, e = r['metal_temperature_K_p05_p50_p95'], r['metal_emissivity_p05_p50_p95']
            writer.writerow([r['variant'], r['best_step'], r['fit']['RMSE_DN'], r['validation']['RMSE_DN'],
                             r['test']['RMSE_DN'], r['validation']['metal_core_RMSE_DN'], r['test']['metal_core_RMSE_DN'],
                             t[1]-273.15 if t else None, e[1] if e else None]+r['cluster_slope_per_K_min_mean_max'])
    (output/'README.md').write_text('Independent full-SH + native-DN attribute readout.\n'
        'Source 3DGS/SH/IR-RGB geometry/mapping are frozen and hashed, no training feedback.\n'
        'T and epsilon are prior-conditioned model estimates, NOT calibrated object-temperature GT.\n'
        'Display mapping is an approximate frozen adapter; native DN anchors the radiometric scale.\n'
        'SH RGB is an auxiliary image-space constraint, never inverted pointwise to DN.\n'
        'Visibility-normalized T/epsilon maps are display summaries, not radiance images.\n'
        'learn_eps_temperature_directional learns region-shared epsilon(T) slopes; other variants keep zero slope.\n'
        'Both endpoint emissivities are bounded across the complete configured temperature interval.\n'
        'Effective metal direction term is bounded, material-region shared,\n'
        'exactly fit-weighted zero mean per point, and is not proof of physical reflection.\n'
        'Variants share initialization, shuffled sampling, budgets and all loss weights.\n'
        'Best selection uses validation balanced DN Huber, including step zero; test only final evaluation.\n'
        'Geometry/SAM3 prior already saw the original train views including probe validation.\n')
    print('COMPLETED: '+str(output), flush=True)
    return str(output)


def plot_history(folder, history):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout='constrained')
    for role in ('fit', 'validation'):
        for ax, key in zip(axes, ('RMSE_DN', 'metal_core_RMSE_DN', 'balanced_DN_Huber')):
            ax.plot([r['step'] for r in history], [r[role][key] for r in history], marker='.', label=role)
            ax.set_title(key); ax.set_xlabel('Probe updates'); ax.grid(alpha=.25); ax.legend()
    fig.savefig(folder/'metrics.png', dpi=150); plt.close(fig)


@torch.no_grad()
def export_previews(folder, field, variant, step, roles, frames, o, g, pipe, rgb_geo, cfg,
                    background, response, supported, deformation, raster, native, predict, display):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from gaussian_renderer import render
    values = field.values(variant, step)
    names = evenly(roles['test'], o.export_views)
    if o.export_views and 'FLIR5715' in roles['test'] and 'FLIR5715' not in names:
        names[-1] = 'FLIR5715'
    as_rgb = lambda x:x.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    for name in names:
        frame = frames[name]; view = frame['view']
        directory = folder/'test'/name; directory.mkdir(parents=True)
        colors = torch.stack([(values['temperature']-o.temp_min)/(o.temp_max-o.temp_min),
                              values['emissivity'], supported.float()], dim=1)
        packed = raster(frame, colors.contiguous(), torch.zeros_like(background))
        acc = raster(frame, torch.ones_like(colors), torch.zeros_like(background))[0]
        valid = (acc >= o.min_accumulation) & (packed[2]/acc.clamp_min(1e-8) >= .95)
        temp = o.temp_min+(o.temp_max-o.temp_min)*packed[0]/acc.clamp_min(1e-8)-273.15
        eps = packed[1]/acc.clamp_min(1e-8)
        slope_colors = values['emissivity_slope_per_K'][:, None].expand(-1, 3).contiguous()
        slope_scale = (o.temp_max-o.temp_min)
        # Rasterize a nonnegative encoding so negative slopes survive the renderer.
        slope_image = raster(frame, (.5+.5*slope_scale*slope_colors).contiguous(), torch.zeros_like(background))[0]
        slope_image = (2*slope_image/acc.clamp_min(1e-8)-1)/slope_scale
        slope_image = slope_image.masked_fill(~valid, float('nan')).cpu().numpy()
        temp, eps = [x.masked_fill(~valid, float('nan')).cpu().numpy() for x in (temp, eps)]
        ir = raster(frame)
        rgb = render(view, g, pipe, background, *deformation(view, rgb_geo, 'rgb'), background.device,
                     feature_set='rgb', detach_geometry=True, disable_atmosphere=True)['render']
        prediction = predict(field, frame, variant, step)
        err = (prediction-frame['target']).abs().masked_fill(~frame['valid'], float('nan'))
        np.savez_compressed(directory/'fields.npz', temperature_C=temp, emissivity=eps,
                            emissivity_slope_per_K=slope_image, temperature_ref_K=o.temperature_ref,
                            field_valid=valid.cpu().numpy(), observed_DN=(frame['target']-response.c['O']).cpu().numpy(),
                            predicted_DN=(prediction-response.c['O']).cpu().numpy(), abs_error_DN=err.cpu().numpy(),
                            native_valid=frame['valid'].cpu().numpy(), metal_core=frame['metal_core'].cpu().numpy())
        fig, axes = plt.subplots(1, 6, figsize=(24, 4.4))
        fig.subplots_adjust(left=.01, right=.99, top=.80, bottom=.16, wspace=.045)
        panels = [as_rgb(view.original_image), as_rgb(ir), temp, eps, as_rgb(view.original_rgb_image), as_rgb(rgb)]
        titles = ['IR-GT', 'IR-Render (frozen SH)', 'T estimate (C)', 'epsilon estimate' if variant != 'fixed_eps_static' else 'epsilon (fixed prior)',
                  'RGB-GT', 'RGB-Render (frozen)']
        for j, ax in enumerate(axes):
            if j in (2, 3):
                cmap = plt.get_cmap('inferno' if j == 2 else 'viridis').copy(); cmap.set_bad('#dddddd')
                limits = (o.temp_min-273.15, o.temp_max-273.15) if j == 2 else (min(.2, o.metal_init-o.metal_half_range), .99)
                im = ax.imshow(np.ma.masked_invalid(panels[j]), cmap=cmap, vmin=limits[0], vmax=limits[1])
                box = ax.get_position(); cax = fig.add_axes([box.x0+.05*box.width, .11, .90*box.width, .02])
                fig.colorbar(im, cax=cax, orientation='horizontal')
            else:
                ax.imshow(panels[j])
            ax.set_title(titles[j], fontsize=12); ax.axis('off')
        fig.suptitle('{} | {} | selected Probe step {}'.format(name, variant, step))
        slope_label = 'Region-shared learned epsilon(T)' if variant == TEMPERATURE_VARIANT else 'Constant emissivity'
        fig.text(.5, .025, 'T/epsilon are prior-conditioned estimates, not ground truth. '+slope_label+'. SH renders are unchanged.', ha='center')
        fig.savefig(directory/'comparison.png', dpi=150); plt.close(fig)
        target = (frame['target']-response.c['O']).cpu().numpy()
        predicted = (prediction-response.c['O']).cpu().numpy()
        error = err.cpu().numpy(); mask = frame['valid'].cpu().numpy()
        lo, hi = np.quantile(target[mask], [.01, .99])
        fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout='constrained')
        for ax, data, label in zip(axes, [np.where(mask, target, np.nan), np.where(mask, predicted, np.nan), error],
                                   ['Original native DN', 'Probe predicted DN', 'Absolute error (DN)']):
            im = ax.imshow(np.ma.masked_invalid(data), cmap='magma' if 'error' in label else 'inferno',
                           vmin=0 if 'error' in label else lo,
                           vmax=max(1, float(np.nanquantile(error, .99))) if 'error' in label else hi)
            ax.set_title(label); ax.axis('off'); fig.colorbar(im, ax=ax, shrink=.75)
        fig.savefig(directory/'dn_diagnostic.png', dpi=150); plt.close(fig)
        # Evaluate display forward on the native predicted signal, then apply
        # the frozen colorize-before-resize adapter. Never colorize splats first.
        y, x = torch.meshgrid(torch.arange(480, device=prediction.device), torch.arange(640, device=prediction.device), indexing='ij')
        xy = torch.stack([2*(x.flatten()+.5)/640-1, 2*(y.flatten()+.5)/480-1], dim=-1)[None]
        normalized = ((prediction-response.c['O'])-frame['meta']['median'])/frame['meta']['span']+.5
        mapped = torch.cat([display(normalized[None], xy[:, start:start+32768], (480, 640))[0]
                            for start in range(0, 640*480, 32768)]).reshape(480, 640, 3).cpu().numpy()
        fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout='constrained')
        for ax, data, title in zip(axes, [as_rgb(view.original_image), as_rgb(ir), mapped],
                                   ['IR GT', 'Frozen full-SH render', 'Probe -> frozen display mapping']):
            ax.imshow(data.clip(0, 1)); ax.set_title(title); ax.axis('off')
        fig.suptitle('Physical readout diagnostic; main SH renderer is unchanged')
        fig.savefig(directory/'display_diagnostic.png', dpi=150); plt.close(fig)


if __name__ == '__main__':
    main()
