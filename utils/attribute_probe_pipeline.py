"""Production orchestration for the tested, detached radiometric readout.

Keep the standalone optimizer as the single implementation. Fit only after
the reconstruction checkpoint is saved; never feed these fields to its loss.
"""
import gc
import hashlib
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
VARIANT = 'learn_eps_temperature_directional'
LEGACY_VARIANT = 'learn_eps_directional'
KIND = 'independent_full_sh_radiometric_attribute_probe'
LIVE_KIND = 'training_full_sh_radiometric_attribute_probe'
CACHE_NAME = 'radiometric_probe.pt'


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(path, value):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def _arguments(model, iteration, opt):
    result = ['-m', str(model), '--iteration', str(iteration), '--variants', VARIANT,
              '--mapping', str((ROOT/opt.phys_attribute_mapping).resolve()),
              '--steps', str(opt.phys_attribute_steps),
              '--views-per-step', str(opt.phys_attribute_views_per_step),
              '--emissivity-lr', str(getattr(opt, 'phys_attribute_emissivity_lr', .003)),
              '--eval-every', str(opt.phys_attribute_eval_every),
              '--log-every', str(opt.phys_attribute_log_every),
              '--temperature-ref', str(opt.phys_probe_temp_ref),
              '--slope-start-fraction', '.75',
              '--slope-lr-scale', str(opt.phys_probe_slope_lr_scale),
              '--slope-prior-weight', str(opt.phys_probe_lambda_slope),
              '--sh-pixels', str(opt.phys_attribute_sh_pixels), '--export-views', '0']
    if float(opt.phys_probe_emissivity_slope) != 0.:
        raise ValueError('DN learnable epsilon(T) requires zero initial phys_probe_emissivity_slope')
    if float(opt.phys_metal_e_init) != -1.:
        result += ['--metal-init', str(opt.phys_metal_e_init)]
    if float(opt.phys_metal_e_half_range) != -1.:
        result += ['--metal-half-range', str(opt.phys_metal_e_half_range)]
    return result


def validate_attribute_inputs(dataset, opt):
    """Fail before training, rather than discovering missing raw DN at 30k."""
    from tools.probe_full_sh_attributes import options
    options(_arguments(dataset.model_path, opt.iterations, opt))
    source = (ROOT/dataset.source_path).resolve()
    path = (ROOT/opt.phys_attribute_mapping).resolve()
    mapping = json.loads(path.read_text())
    if (mapping.get('kind') != 'approximate_FLIR_display_adapter'
            or (ROOT/mapping['settings']['scene']).resolve() != source
            or mapping.get('test_used_for_fitting_or_selection') is not False
            or mapping.get('order') != 'colorize_then_resize'):
        raise ValueError('Attribute Probe requires the matching frozen DN/display mapping')
    if not dataset.eval or dataset.data_branch != 'rgbt' or dataset.is_6dof:
        raise ValueError('Tested attribute readout requires additive RGBT geometry and --eval')
    if getattr(dataset, 'resolution', -1) not in (-1, 1, 640):
        raise ValueError('Frozen attribute mapping requires the loaded 640x480 training grid')
    if getattr(opt, 'ir_atmosphere', False) or getattr(opt, 'ir_eval_temporal_ensemble', False):
        raise ValueError('Tested attribute readout does not support atmosphere/temporal ensemble corrections')
    from PIL import Image
    from tools.fit_flir_display_mapping import extract_flir
    names = sorted(mapping['fit_frames']+mapping['validation_frames'])
    test_names = sorted(p.stem for p in (source/'thermal/test').glob('*.jpg'))
    if not names or not test_names or len(set(names+test_names)) != len(names)+len(test_names):
        raise ValueError('Invalid fit/validation/test names for attribute Probe')
    response = None
    for name in names+test_names:
        _, _, meta = extract_flir(source/'raw_images'/(name+'.jpg'), mapping['png_byte_order'])
        if response is None:
            response = meta['planck']
        if response != meta['planck']:
            raise ValueError('Mixed camera responses unsupported')
        split = 'test' if name in test_names else 'train'
        for branch in ('rgb', 'thermal'):
            with Image.open(source/branch/split/(name+'.jpg')) as image:
                if image.size != (640, 480):
                    raise ValueError('Frozen attribute adapter requires original 640x480 images')
    print('[ATTRIBUTE PROBE] inputs verified; independent fit budget={} x {}; '
          '{}; learnable regional slope; epsilon0_raw_lr={} slope_raw_lr={}; feedback_weight={}'.format(
              opt.phys_attribute_steps, opt.phys_attribute_views_per_step, VARIANT,
              float(getattr(opt, 'phys_attribute_emissivity_lr', .003)),
              float(getattr(opt, 'phys_attribute_emissivity_lr', .003))*opt.phys_probe_slope_lr_scale,
              float(opt.phys_aux_mse_max_weight)), flush=True)


def _validate_emissivity_fields(cache):
    """Validate epsilon(T) over the full interval, retaining schema-1 controls."""
    fields, state, options = cache['fields'], cache['state_dict'], cache['options']
    t, e = fields['temperature'].reshape(-1), fields['emissivity'].reshape(-1)
    ids = state['ids'].long().reshape(-1)
    low, high = state['low'].reshape(-1), state['high'].reshape(-1)
    if (len(ids) != len(t) or len(e) != len(t) or not len(ids)
            or ids.min() < 0 or ids.max() >= len(low) or len(low) != len(high)
            or not torch.isfinite(low).all() or not torch.isfinite(high).all()
            or not ((low > 0) & (low < high) & (high <= 1)).all()
            or not torch.isfinite(t).all() or not torch.isfinite(e).all()):
        raise ValueError('Invalid temperature/emissivity fields or material bounds')
    t_min, t_max = float(options['temp_min']), float(options['temp_max'])
    if not (0 < t_min < t_max) or not ((t >= t_min-1e-4) & (t <= t_max+1e-4)).all():
        raise ValueError('Temperature fields lie outside the fitted interval')
    if cache.get('emissivity_model', 'constant') == 'linear_temperature':
        if cache.get('schema_version') != 2 or cache.get('variant') != VARIANT or cache.get('slope_learnable') is not True:
            raise ValueError('Invalid learnable epsilon(T) cache schema')
        ref = float(cache['temperature_ref_K'])
        eps0 = fields['cluster_emissivity'].reshape(-1)
        slope = fields['cluster_slope_per_K'].reshape(-1)
        if (not t_min <= ref <= t_max or len(eps0) != len(low) or len(slope) != len(low)
                or not torch.isfinite(eps0).all() or not torch.isfinite(slope).all()):
            raise ValueError('Invalid epsilon(T) coefficients')
        endpoints = torch.stack([eps0+slope*(temp-ref) for temp in (t_min, t_max)])
        expected = eps0[ids]+slope[ids]*(t-ref)
        if (not ((endpoints >= low-1e-6) & (endpoints <= high+1e-6)).all()
                or not torch.allclose(e, expected, rtol=1e-6, atol=1e-6)
                or not torch.allclose(fields['emissivity_ref'].reshape(-1), eps0[ids], rtol=1e-6, atol=1e-6)
                or not torch.allclose(fields['emissivity_slope_per_K'].reshape(-1), slope[ids], rtol=1e-6, atol=1e-9)):
            raise ValueError('Inconsistent or out-of-bounds epsilon(T) fields')
        return slope[ids]
    if cache.get('emissivity_model', 'constant') != 'constant' or cache.get('slope_per_K') != 0.:
        raise ValueError('Unknown emissivity model or nonzero slope in constant cache')
    if not ((e >= low[ids]-1e-6) & (e <= high[ids]+1e-6)).all():
        raise ValueError('Constant emissivity is outside material bounds')
    return torch.zeros_like(e)


def attach_attribute_result(model_path, iteration, result_path):
    """Bind a verified standalone result to exactly its source checkpoint."""
    model, result = Path(model_path).resolve(), Path(result_path).resolve()
    summary = json.loads((result/'summary.json').read_text())
    if (summary.get('kind') != KIND or not summary.get('source_assets_unchanged')
            or Path(summary['source_model']).resolve() != model
            or int(summary['source_iteration']) != iteration):
        raise ValueError('Attribute result provenance does not match the checkpoint')
    selected = next((r for r in summary['variants'] if r['variant'] == VARIANT), None)
    if selected is None:
        selected = next((r for r in summary['variants'] if r['variant'] == LEGACY_VARIANT), None)
    if selected is None:
        raise ValueError('Missing supported directional variant')
    variant = selected['variant']
    source_hashes = summary['source_sha256_before']
    # The code may have been integrated since an independent run. Its source
    # MODEL (including cfg, IR/RGB geometry and observation cache) must match.
    for name, expected in source_hashes.items():
        if model in Path(name).parents and _sha(name) != expected:
            raise ValueError('Source model asset changed after attribute fit: '+name)
    checkpoint = model/'point_cloud'/('iteration_%d'%iteration)
    ply = checkpoint/'point_cloud.ply'
    cache = torch.load(result/variant/'probe_best.pt', map_location='cpu')
    if (cache.get('kind') != KIND or cache.get('variant') != variant
            or cache.get('source_iteration') != iteration
            or Path(cache['source_model']).resolve() != model
            or int(cache['step']) != int(selected['best_step'])):
        raise ValueError('Invalid best attribute cache')
    _validate_emissivity_fields(cache)
    temperature = cache['fields']['temperature'].reshape(-1)
    epsilon = cache['fields']['emissivity'].reshape(-1)
    n = len(temperature)
    with ply.open('rb') as stream:
        for line in stream:
            if line.startswith(b'element vertex '):
                if int(line.split()[-1]) != n:
                    raise ValueError('Attribute/Gaussian point count mismatch')
                break
        else:
            raise ValueError('Missing vertex count in PLY')
    ids = cache['state_dict']['ids'].long()
    low, high = cache['state_dict']['low'][ids], cache['state_dict']['high'][ids]
    if (len(epsilon) != n or not torch.isfinite(temperature).all() or not torch.isfinite(epsilon).all()
            or not ((epsilon >= low-1e-6) & (epsilon <= high+1e-6)).all()):
        raise ValueError('Invalid physical-field values')
    geometry = {folder:_sha(model/folder/('iteration_%d'%iteration)/'geometry.pth')
                for folder in ('Geometry', 'GeometryRGB')}
    cache['main_integration'] = dict(kind='detached_post_training_attribute_readout',
        point_cloud_sha256=_sha(ply), geometry_sha256=geometry,
        fit_output=str(result), selected_step=int(cache['step']), main_loss_feedback=False,
        validation=selected['validation'], test=selected['test'])
    target = checkpoint/CACHE_NAME
    if target.exists():
        old = torch.load(target, map_location='cpu')
        if old.get('main_integration', {}).get('fit_output') == str(result):
            print('[ATTRIBUTE PROBE] already attached: '+str(target), flush=True)
            return target
        raise FileExistsError('An attribute cache already exists; keep it or use a new model output: '+str(target))
    temporary = target.with_suffix('.pt.tmp')
    torch.save(cache, temporary); temporary.replace(target)
    pointer = model/'phys_probe'/('iteration_%d'%iteration)/'attribute_fit.json'
    pointer.parent.mkdir(parents=True, exist_ok=True)
    _json(pointer, dict(cache=str(target), fit_output=str(result), variant=variant,
                        selected_step=int(cache['step']), slope_per_K=cache['slope_per_K'],
                        slope_learnable=cache.get('slope_learnable', False),
                        emissivity_model=cache.get('emissivity_model', 'constant'),
                        temperature_ref_K=cache.get('temperature_ref_K'),
                        cluster_slope_per_K_min_mean_max=selected.get('cluster_slope_per_K_min_mean_max'),
                        temperature_ground_truth=False, emissivity_ground_truth=False,
                        main_loss_feedback=False, validation=selected['validation'], test=selected['test']))
    with (model/'result.txt').open('a') as file:
        file.write('[ATTRIBUTE PROBE] attached {} step={} cache={}\n'.format(variant, cache['step'], target))
    print('[ATTRIBUTE PROBE] attached '+str(target), flush=True)
    return target


def fit_final_attributes(model_path, opt):
    from tools.probe_full_sh_attributes import main as fit
    model = (ROOT/model_path).resolve()
    iteration = int(opt.iterations)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print('[ATTRIBUTE PROBE] reconstruction finished; fitting frozen final checkpoint.', flush=True)
    result = fit(_arguments(model, iteration, opt))
    return attach_attribute_result(model, iteration, result)


def select_attribute_iteration(model_path, requested):
    if requested > 0:
        return int(requested)
    if requested != -1:
        raise ValueError('physical_fields requires a positive iteration or -1')
    available = [int(p.parent.name.split('_')[-1])
                 for p in (Path(model_path)/'point_cloud').glob('iteration_*/'+CACHE_NAME)]
    if not available:
        raise FileNotFoundError('This model has no fitted '+CACHE_NAME)
    return max(available)


@torch.no_grad()
def render_attribute_fields(scene, gaussians, geo_model, pipeline, model_path, opt, iteration,
                            device, is_6dof, load2gpu_on_the_fly, rgb_geo_model=None,
                            skip_train=True, skip_test=False, background=None, is_blender=False,
                            color_renderer=None, temporal_pose_model=None, scene_extent=1.0,
                            temperature_range=None):
    """Render cached fields only; this function never refits or changes SH."""
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    from torchvision.utils import save_image
    from gaussian_renderer import render
    from train import apply_temporal_pose_delta, compute_geo_fields, estimate_temporal_jitter_interval
    model = Path(model_path).resolve()
    folder = model/'point_cloud'/('iteration_%d'%iteration)
    path = folder/CACHE_NAME
    if not path.exists():
        raise FileNotFoundError('No fitted radiometric fields for iteration {}. '
                                'Use the final fitted checkpoint (usually --iteration 30000), '
                                'not the best-PSNR mirror. Expected {}'.format(iteration, path))
    cache = torch.load(path, map_location='cpu')
    slope = _validate_emissivity_fields(cache).to(device)
    binding = cache.get('main_integration', {})
    if (cache.get('kind') not in (KIND, LIVE_KIND) or cache.get('source_iteration') != iteration
            or binding.get('point_cloud_sha256') != _sha(folder/'point_cloud.ply')):
        raise ValueError('Attribute cache does not match this point cloud')
    for name, expected in binding['geometry_sha256'].items():
        if _sha(model/name/('iteration_%d'%iteration)/'geometry.pth') != expected:
            raise ValueError('Geometry changed after attribute fit: '+name)
    n = len(gaussians.get_xyz)
    t = cache['fields']['temperature'].reshape(-1).to(device)
    e = cache['fields']['emissivity'].reshape(-1).to(device)
    support = cache['supported'].reshape(-1).to(device)
    if len(t) != n or len(e) != n or len(support) != n:
        raise ValueError('Field length differs from Gaussian count')
    options = cache['options']
    minimum = options['min_accumulation']
    lo, hi = options['temp_min'], options['temp_max']
    if temperature_range is not None:
        temperature_range = tuple(float(value) for value in temperature_range)
        if (len(temperature_range) != 2 or not np.isfinite(temperature_range).all()
                or temperature_range[0] >= temperature_range[1]):
            raise ValueError('temperature_range requires finite MIN_C < MAX_C')
    epsilon_limits = (float(cache['state_dict']['low'].min()),
                      float(cache['state_dict']['high'].max()))
    panel_order = ['IR GT', 'IR Render', 'RGB GT', 'RGB Render',
                   'T Distribution', 'epsilon Distribution']
    panel_files = ['ir_gt.png', 'ir_render.png', 'rgb_gt.png', 'rgb_render.png',
                   'temperature.png', 'emissivity.png']
    if background is None:
        raise ValueError('Pass the source checkpoint background to the field renderer')
    black = torch.zeros(3, device=device)
    interval = estimate_temporal_jitter_interval(scene.getTrainCameras(), 'fid_median')
    def deformation(view, network, feature):
        delta, rotation, scaling = compute_geo_fields(
            view, gaussians, network, pipeline, background, opt, iteration,
            lambda _:0., interval, device, is_blender, feature_set=feature, eval_mode=True)
        delta, _, _, _ = apply_temporal_pose_delta(
            delta, gaussians, view.fid, temporal_pose_model, opt, iteration, scene_extent)
        return delta, rotation, scaling
    rows = []
    for split, views, skip in [('train', scene.getTrainCameras(), skip_train),
                               ('test', scene.getTestCameras(), skip_test)]:
        if skip:
            continue
        root = model/split/('physical_%d'%iteration)
        if not views:
            raise ValueError('No '+split+' cameras available for physical-field rendering')
        if any(not getattr(view, 'is_rgbt', False) or view.original_rgb_image is None for view in views):
            raise ValueError('Six-panel export requires paired RGB and IR cameras')
        root.mkdir(parents=True, exist_ok=True)
        for view in views:
            original_device = view.world_view_transform.device
            view.load2device(device)
            try:
                deform = deformation(view, geo_model, 'thermal')
                def raster(colors=None):
                    return render(view, gaussians, pipeline, background if colors is None else black,
                                  *deform, device, is_6dof=is_6dof, feature_set='thermal',
                                  override_color=colors, detach_geometry=True, disable_atmosphere=True)['render']
                packed = raster(torch.stack([(t-lo)/(hi-lo), e, support.float()], -1).contiguous())
                acc = raster(torch.ones(n, 3, device=device))[0]
                valid = (acc >= minimum) & (packed[2]/acc.clamp_min(1e-8) >= .95)
                temp = lo+(hi-lo)*packed[0]/acc.clamp_min(1e-8)-273.15
                eps = packed[1]/acc.clamp_min(1e-8)
                span = hi-lo
                encoded_slope = (.5+.5*span*slope)[:, None].expand(-1, 3).contiguous()
                slope_map = (2*raster(encoded_slope)[0]/acc.clamp_min(1e-8)-1)/span
                slope_map = slope_map.masked_fill(~valid, float('nan')).cpu().numpy()
                temp, eps = [x.masked_fill(~valid, float('nan')).cpu().numpy() for x in (temp, eps)]
                if color_renderer is not None:
                    ir = color_renderer(view, 'thermal')
                    rgb = color_renderer(view, 'rgb')
                else:
                    ir = raster()
                    rgb_deform = deformation(view, rgb_geo_model or geo_model, 'rgb')
                    rgb = render(view, gaussians, pipeline, background, *rgb_deform, device,
                                 is_6dof=is_6dof, feature_set='rgb', detach_geometry=True, disable_atmosphere=True)['render']
                output = root/view.image_name; output.mkdir(exist_ok=True)
                for image, filename in zip(
                        [view.original_image[:3], ir, view.original_rgb_image[:3], rgb], panel_files[:4]):
                    save_image(image, str(output/filename))
                Image.fromarray(valid.cpu().numpy().astype(np.uint8)*255).save(output/'valid_mask.png')
                np.savez_compressed(output/'fields.npz', temperature_C=temp, emissivity=eps,
                                    emissivity_slope_per_K=slope_map,
                                    temperature_ref_K=cache.get('temperature_ref_K', 300.),
                                    field_valid=valid.cpu().numpy(), accumulation=acc.cpu().numpy())
                rows.append(dict(view=view.image_name, split=split, valid_fraction=float(valid.float().mean()),
                                 temperature_C_min_max=[float(np.nanmin(temp)), float(np.nanmax(temp))] if np.isfinite(temp).any() else None,
                                 temperature_C_p05_p50_p95=np.nanquantile(temp, [.05, .5, .95]).tolist() if np.isfinite(temp).any() else None))
            finally:
                view.load2device(original_device)

        # Choose one colorbar from all rendered views. Do not hide small metal
        # regions with percentiles or stretch each camera independently.
        split_rows = [row for row in rows if row['split'] == split]
        ranges = [row['temperature_C_min_max'] for row in split_rows
                  if row['temperature_C_min_max'] is not None]
        if not ranges:
            raise ValueError('No supported temperature pixels in '+split)
        actual_temperature = (min(value[0] for value in ranges), max(value[1] for value in ranges))
        temperature_limits = temperature_range or (
            float(5*np.floor(actual_temperature[0]/5)),
            float(5*np.ceil(actual_temperature[1]/5)),
        )
        if temperature_limits[0] == temperature_limits[1]:
            temperature_limits = (temperature_limits[0]-5, temperature_limits[1]+5)
        below = actual_temperature[0] < temperature_limits[0]
        above = actual_temperature[1] > temperature_limits[1]
        temperature_extend = 'both' if below and above else 'min' if below else 'max' if above else 'neither'
        print('[PHYSICAL FIELDS] {} temperature {:.2f}..{:.2f} C; shared colorbar {:.1f}..{:.1f} C'.format(
            split, *actual_temperature, *temperature_limits), flush=True)
        for row in split_rows:
            output = root/row['view']
            with np.load(output/'fields.npz') as fields:
                temp, eps = fields['temperature_C'], fields['emissivity']
            panels = []
            for filename in panel_files[:4]:
                with Image.open(output/filename) as image:
                    panels.append(np.array(image.convert('RGB')))
            panels += [temp, eps]
            fig, axes = plt.subplots(1, 6, figsize=(24, 4.4))
            fig.subplots_adjust(left=.01, right=.99, top=.80, bottom=.16, wspace=.045)
            titles = panel_order[:4]+['T Distribution (C)', r'$\epsilon$ Distribution']
            for i, (ax, data, title) in enumerate(zip(axes, panels, titles)):
                if i in (4, 5):
                    cmap = plt.get_cmap('inferno' if i == 4 else 'viridis').copy(); cmap.set_bad('#dddddd')
                    limits = temperature_limits if i == 4 else epsilon_limits
                    im = ax.imshow(np.ma.masked_invalid(data), cmap=cmap, vmin=limits[0], vmax=limits[1])
                    box = ax.get_position(); cax = fig.add_axes([box.x0+.05*box.width, .11, .90*box.width, .02])
                    fig.colorbar(im, cax=cax, orientation='horizontal',
                                 extend=temperature_extend if i == 4 else 'neither')
                else:
                    ax.imshow(data)
                ax.set_title(title, fontsize=12); ax.axis('off')
            fig.suptitle('{} | checkpoint {} | {} | Probe step {}'.format(row['view'], iteration, cache['variant'], cache['step']))
            slope_label = 'Region-shared learned epsilon(T)' if cache.get('slope_learnable') else 'Constant emissivity'
            fig.text(.5, .025, 'T/epsilon are estimated fields. Gray: insufficient support. '+slope_label+'.', ha='center')
            fig.savefig(output/'comparison.png', dpi=150); plt.close(fig)
            for data, name, cmap_name, lower, upper in [(temp, 'temperature', 'inferno', *temperature_limits),
                                                      (eps, 'emissivity', 'viridis', *epsilon_limits)]:
                fig, ax = plt.subplots(figsize=(6.5, 5))
                cmap = plt.get_cmap(cmap_name).copy(); cmap.set_bad('#dddddd')
                im = ax.imshow(np.ma.masked_invalid(data), cmap=cmap, vmin=lower, vmax=upper)
                ax.set_title(row['view']+' | '+name+(' (C)' if name == 'temperature' else ''))
                ax.axis('off'); fig.colorbar(im, ax=ax, shrink=.8,
                                            extend=temperature_extend if name == 'temperature' else 'neither')
                fig.savefig(output/(name+'.png'), dpi=150, bbox_inches='tight'); plt.close(fig)
            print('[PHYSICAL FIELDS] '+str(output/'comparison.png'), flush=True)
        _json(root/'summary.json', dict(source_cache=str(path), source_iteration=iteration,
                                        panel_order=panel_order, panel_files=panel_files,
                                        temperature_C_limits=list(temperature_limits),
                                        actual_temperature_C_min_max=list(actual_temperature),
                                        temperature_colorbar_extend=temperature_extend,
                                        temperature_colorbar_mode='manual' if temperature_range is not None else 'shared_actual_range',
                                        emissivity_limits=list(epsilon_limits),
                                        invalid_color='#dddddd', temperature_unit='C',
                                        emissivity_model=cache.get('emissivity_model', 'constant'),
                                        slope_learnable=cache.get('slope_learnable', False),
                                        temperature_ground_truth=False, emissivity_ground_truth=False,
                                        main_loss_feedback=bool(binding.get('main_loss_feedback', False)),
                                        views=[r for r in rows if r['split'] == split]))
    return rows
