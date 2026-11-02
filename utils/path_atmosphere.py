"""Optional, effective band-averaged IR atmosphere. No test images enter the Probe.

Distances use reconstruction units, not metres. For uncalibrated thermal PNGs,
air/source are image-domain proxies and mu is not a measured absorption coefficient.
"""
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F


def range_from_depth(depth_sum, acc, fovx, fovy):
    """Native rasterizer depth is sum(w*z); divide coverage before ray conversion."""
    depth_sum, acc = depth_sum.detach(), acc.detach()
    h, w = depth_sum.shape[-2:]
    # Matches CUDA ndc2Pix: ((ndc + 1) * size - 1) / 2.
    x = (2 * (torch.arange(w, device=acc.device, dtype=acc.dtype) + .5) / w - 1) * math.tan(fovx / 2)
    y = (2 * (torch.arange(h, device=acc.device, dtype=acc.dtype) + .5) / h - 1) * math.tan(fovy / 2)
    z = depth_sum / acc.clamp_min(1e-6)
    distance = z * torch.sqrt(1 + y[:, None].square() + x[None, :].square())
    return torch.where((acc > 1e-6) & torch.isfinite(distance) & (distance > 0), distance, 0)


def apply_atmosphere(image, depth_sum, acc, background, fovx, fovy, mu, air):
    """Attenuate covered foreground; retain the renderer's original background.

    Geometry/coverage are constants only in this extra path. The image keeps its
    original rasterizer gradients, including geometry, SH and opacity gradients.
    """
    if mu == 0:
        return image
    acc = acc.detach().clamp(0, 1)
    distance = range_from_depth(depth_sum, acc, fovx, fovy)
    tau = torch.exp(-float(mu) * distance)
    bg = (1 - acc) * background[:, None, None]
    return tau * (image - bg) + acc * (1 - tau) * float(air) + bg


@torch.no_grad()
def fit_profile(distances, observed, valid, air, distance_ref, max_beta=.3,
                min_span=.02, min_tracks=64, min_gain=.001, previous_beta=0.):
    """Profile out one view-independent source per track, then grid-fit beta=mu*d_ref.

    Shapes: distances/valid [view, track], observed [view, track, channel].
    Every fourth *training* view is held out to reject spurious corrections.
    A global grid includes the exact zero model; no softplus-at-zero dead start.
    """
    v, n = distances.shape
    if v < 6 or distance_ref <= 0:
        return {"accepted": False, "reason": "insufficient_views_or_distance"}
    valid = valid & torch.isfinite(distances) & (distances > 0) & torch.isfinite(observed).all(-1)
    holdout = torch.arange(v, device=distances.device).remainder(4) == 3
    fit_mask = valid & ~holdout[:, None]
    check_mask = valid & holdout[:, None]
    lo = distances.masked_fill(~valid, float('inf')).amin(0)
    hi = distances.masked_fill(~valid, -float('inf')).amax(0)
    mean = (distances.masked_fill(~valid, 0).sum(0) / valid.sum(0).clamp_min(1)).clamp_min(1e-6)
    contrast = ((observed - air).abs().mean(-1).masked_fill(~valid, 0).sum(0) / valid.sum(0).clamp_min(1))
    keep = ((fit_mask.sum(0) >= 3) & (check_mask.sum(0) >= 1)
            & ((hi - lo) / mean >= min_span) & (contrast >= .03))
    count = int(keep.sum())
    if count < min_tracks:
        return {"accepted": False, "reason": "insufficient_supported_tracks", "tracks": count}
    d = distances[:, keep] / distance_ref
    y = observed[:, keep]
    fm, cm = fit_mask[:, keep], check_mask[:, keep]
    # Invalid observations never contribute, including NaNs in rejected samples.
    d = torch.where(valid[:, keep], d, torch.zeros_like(d))
    y = torch.where(valid[:, keep, None], y, torch.full_like(y, air))
    candidates = torch.cat([torch.linspace(0, max_beta, 61, device=d.device),
                            d.new_tensor([max(0., min(max_beta, previous_beta))])]).unique(sorted=True)
    best, zero = None, None
    for beta in candidates:
        tau = torch.exp(-beta * d)
        w = fm.to(d.dtype)
        source = air + ((tau * w)[..., None] * (y - air)).sum(0) / (tau.square() * w).sum(0).clamp_min(1e-8)[:, None]
        source = source.clamp(0, 1)
        pred = air + tau[..., None] * (source - air)
        residual = F.smooth_l1_loss(pred, y, beta=.02, reduction='none').mean(-1)
        # Average within each track first, so frequently visible surfaces cannot dominate.
        train = ((residual * fm).sum(0) / fm.sum(0)).mean()
        check = ((residual * cm).sum(0) / cm.sum(0)).mean()
        row = {"beta": float(beta), "train_loss": float(train), "holdout_loss": float(check)}
        objective = float(train) + 1e-6 * float(beta)**2
        if zero is None: zero = row
        if best is None or objective < best[0]: best = (objective, row)
    row = best[1]
    gain = (zero['holdout_loss'] - row['holdout_loss']) / max(zero['holdout_loss'], 1e-8)
    return {**row, "tracks": count, "zero_holdout_loss": zero['holdout_loss'],
            "relative_holdout_gain": gain, "accepted": row['beta'] > 0 and gain >= min_gain,
            "reason": "fit" if row['beta'] > 0 and gain >= min_gain else "no_supported_gain"}


class PathAtmosphere:
    """Detached scalar Probe state; main training never optimizes mu or air."""
    def __init__(self, opt):
        self.opt = opt
        self.iteration = 0
        self.previous_mu = self.target_mu = 0.
        self.updated_at = 0
        self.distance_ref = None
        if opt.ir_atmosphere_air < 0 and opt.ir_atmosphere_air != -1:
            raise ValueError('ir_atmosphere_air must be -1 (train quantile) or in [0,1]')
        self.air = None if opt.ir_atmosphere_air == -1 else float(opt.ir_atmosphere_air)
        if self.air is not None and not 0 <= self.air <= 1:
            raise ValueError('ir_atmosphere_air must be -1 (train quantile) or in [0,1]')
        if not 0 < opt.ir_atmosphere_max_beta <= 2:
            raise ValueError('ir_atmosphere_max_beta must be in (0,2]')
        if opt.ir_atmosphere_probe_views < 6 or opt.ir_atmosphere_probe_every <= 0:
            raise ValueError('Atmosphere Probe needs >=6 train views and a positive interval')
        if opt.ir_atmosphere_ramp_iters < 0 or not 0 <= opt.ir_atmosphere_air_quantile <= 1:
            raise ValueError('Invalid atmosphere ramp or air quantile')
        if (opt.ir_atmosphere_probe_points < 1 or opt.ir_atmosphere_min_tracks < 1
                or opt.ir_atmosphere_probe_scale < 1 or opt.ir_atmosphere_start_iter < 0):
            raise ValueError('Invalid atmosphere sampling configuration')
        if (not 0 < opt.ir_atmosphere_min_acc <= 1 or opt.ir_atmosphere_min_distance_span < 0
                or not 0 <= opt.ir_atmosphere_min_gain < 1):
            raise ValueError('Invalid atmosphere support thresholds')

    @property
    def mu(self):
        if self.opt.ir_atmosphere_ramp_iters == 0:
            return self.target_mu
        ramp = max(1, self.opt.ir_atmosphere_ramp_iters)
        t = min(1., max(0., (self.iteration - self.updated_at) / ramp))
        return (1-t)*self.previous_mu + t*self.target_mu

    def save(self, model_path, checkpoint):
        path = Path(model_path)/'PathAtmosphere'/f'iteration_{checkpoint}'/'state.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {k: v for k, v in vars(self).items() if k != 'opt'}
        path.write_text(json.dumps({"version": 1, **state}, indent=2)+'\n')

    def load(self, model_path, checkpoint):
        path = Path(model_path)/'PathAtmosphere'/f'iteration_{checkpoint}'/'state.json'
        data = json.loads(path.read_text())
        for k in ('iteration', 'previous_mu', 'target_mu', 'updated_at', 'distance_ref', 'air'):
            setattr(self, k, data[k])
        if any(not math.isfinite(v) or v < 0 for v in (self.previous_mu, self.target_mu)):
            raise ValueError('Atmosphere checkpoint contains an invalid mu')

    @torch.no_grad()
    def probe(self, gaussians, scene, pipe, background, deformation, is_6dof, output):
        from gaussian_renderer import render
        from utils.rigid_utils import from_homogenous, to_homogenous
        opt = self.opt
        # Fixed selection does not consume the training RNG or shuffle/pop camera lists.
        cameras = sorted(scene.getTrainCameras(), key=lambda c: c.image_name)
        camera_ids = torch.linspace(0, len(cameras)-1, min(len(cameras), opt.ir_atmosphere_probe_views)).round().long().unique().tolist()
        anchors = torch.where(gaussians.get_opacity[:, 0] > .1)[0]
        if len(anchors) > opt.ir_atmosphere_probe_points:
            ids = torch.linspace(0, len(anchors)-1, opt.ir_atmosphere_probe_points, device=anchors.device).round().long()
            anchors = anchors[ids]
        if len(camera_ids) < 6 or not len(anchors):
            self.record(output, {'iteration': self.iteration, 'accepted': False, 'reason': 'insufficient_cameras_or_anchors', 'mu': self.mu})
            return
        distances, observations, masks, air_samples = [], [], [], []
        for index in camera_ids:
            camera = cameras[index]
            original_device = camera.original_image.device
            if original_device != gaussians.get_xyz.device:
                camera.load2device(gaussians.get_xyz.device)
            try:
                dx, dr, ds = deformation(camera)
                pkg = render(camera, gaussians, pipe, background, dx, dr, ds,
                             gaussians.get_xyz.device, is_6dof, feature_set='thermal',
                             scale_factor=opt.ir_atmosphere_probe_scale,
                             atmosphere_aux=True, disable_atmosphere=True)
                if is_6dof and torch.is_tensor(dx):
                    xyz = from_homogenous(torch.bmm(dx, to_homogenous(gaussians.get_xyz).unsqueeze(-1)).squeeze(-1))[anchors]
                else:
                    xyz = (gaussians.get_xyz + dx)[anchors]
                homogeneous = torch.cat([xyz, torch.ones_like(xyz[:, :1])], -1)
                clip = homogeneous @ camera.full_proj_transform
                ndc = clip[:, :2] / clip[:, 3:4].clamp_min(1e-6)
                z = (homogeneous @ camera.world_view_transform)[:, 2]
                grid = ndc.view(1, 1, -1, 2)
                def sample(image):
                    return F.grid_sample(image[None], grid, align_corners=False, padding_mode='zeros')[0, :, 0].T
                acc = sample(pkg['acc'])[:, 0]
                z_depth = sample(pkg['depth'] / pkg['acc'].clamp_min(1e-6))[:, 0]
                distance = sample(pkg['distance'])[:, 0]
                gt = F.interpolate(camera.original_image[None, :3], size=pkg['depth'].shape[-2:], mode='area')
                mean = F.avg_pool2d(gt, 3, 1, 1)
                variance = (F.avg_pool2d(gt.square(), 3, 1, 1) - mean.square()).clamp_min(0)
                observed = sample(mean[0])
                local_std = sample(variance[0]).mean(-1).sqrt()
                valid = ((z > .01) & (ndc.abs().amax(-1) < .97)
                         & (acc >= opt.ir_atmosphere_min_acc)
                         & ((z-z_depth).abs() <= .03*z_depth.clamp_min(.01))
                         & (local_std <= .05))
                distances.append(distance)
                observations.append(observed)
                masks.append(valid)
                air_samples.append(gt.mean(1).flatten()[::16])
            finally:
                if original_device != gaussians.get_xyz.device:
                    camera.load2device(original_device)
        distances, observed, valid = torch.stack(distances), torch.stack(observations), torch.stack(masks)
        # Fix the image-domain air proxy once; do not let it trade off freely with mu.
        if self.air is None:
            self.air = float(torch.quantile(torch.cat(air_samples), opt.ir_atmosphere_air_quantile))
        usable = distances[valid]
        if not len(usable):
            self.record(output, {'iteration': self.iteration, 'accepted': False, 'reason': 'no_visible_surface_tracks', 'mu': self.mu, 'air': self.air})
            return
        if self.distance_ref is None:
            self.distance_ref = float(usable.median())
        result = fit_profile(distances, observed, valid, self.air, self.distance_ref,
                             opt.ir_atmosphere_max_beta, opt.ir_atmosphere_min_distance_span,
                             opt.ir_atmosphere_min_tracks, opt.ir_atmosphere_min_gain,
                             self.target_mu * self.distance_ref)
        current = self.mu
        # An unsupported fit returns towards the exact zero model, rather than
        # preserving an obsolete nonzero correction after geometry changes.
        self.previous_mu = current
        self.target_mu = result['beta']/self.distance_ref if result['accepted'] else 0.
        self.updated_at = self.iteration
        self.record(output, {**result, 'iteration': self.iteration, 'mu': current,
                             'target_mu': self.target_mu, 'distance_ref': self.distance_ref,
                             'air': self.air, 'views': len(camera_ids), 'anchors': len(anchors),
                             'domain': 'effective_image_proxy_per_reconstruction_unit'})

    def record(self, output, row):
        path = Path(output)/'path_atmosphere.jsonl'
        with path.open('a') as stream:
            stream.write(json.dumps(row)+'\n')
        print('[PathAtmosphere] '+json.dumps(row))
