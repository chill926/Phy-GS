"""Shared DN response, frozen display mapping and view sampling for attribute Probe.

Production attribute fitting is in tools/.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def read_config(model, device):
    from arguments import ModelParams, OptimizationParams, PipelineParams
    p = argparse.ArgumentParser(add_help=False)
    ModelParams(p, device); OptimizationParams(p); PipelineParams(p)
    values = vars(p.parse_args([]))
    node = ast.parse((model / 'cfg_args').read_text(), mode='eval').body
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.func.id != 'Namespace' or node.args:
        raise ValueError('Expected saved Namespace cfg_args')
    for item in node.keywords:
        if item.arg is None:
            raise ValueError('Expanded cfg_args unsupported')
        v = item.value
        if isinstance(v, ast.Call):
            constructor = v.func
            is_device = isinstance(constructor, ast.Name) and constructor.id == 'device'
            is_torch_device = (isinstance(constructor, ast.Attribute) and constructor.attr == 'device'
                               and isinstance(constructor.value, ast.Name) and constructor.value.id == 'torch')
            if not (is_device or is_torch_device):
                raise ValueError('Unsupported constructor in cfg_args')
            values[item.arg] = torch.device(*[ast.literal_eval(a) for a in v.args],
                                           **{k.arg: ast.literal_eval(k.value) for k in v.keywords})
        else:
            values[item.arg] = ast.literal_eval(v)
    values.update(model_path=str(model), load_model_path='', load_iteration=0,
                  data_device=device, load2gpu_on_the_fly=True)
    values['source_path'] = str((ROOT / values['source_path']).resolve())
    cfg = argparse.Namespace(**values)
    if not cfg.eval or cfg.data_branch != 'rgbt' or cfg.is_6dof:
        raise ValueError('Requires an additive-geometry RGBT checkpoint with eval split')
    forbidden = ('ir_temporal_pose', 'ir_temporal_affine', 'ir_eval_temporal_ensemble',
                 'ir_atmosphere', 'phys_eval_with_render', 'phys_main_render')
    if any(getattr(cfg, k, False) for k in forbidden):
        raise ValueError('Unsupported extra correction: ' + ', '.join(k for k in forbidden if getattr(cfg, k, False)))
    return cfg


class CameraResponse:
    def __init__(self, coefficients):
        self.c = dict(coefficients)
        for key in ('R1', 'R2', 'B', 'F', 'O'):
            if key not in self.c or not np.isfinite(self.c[key]):
                raise ValueError('Missing/nonfinite camera response: ' + key)
        if min(self.c[k] for k in ('R1', 'R2', 'B')) <= 0:
            raise ValueError('Invalid positive response coefficients')

    def forward(self, temperature):
        c = self.c
        return c['R1'] / (c['R2'] * (torch.exp(c['B'] / temperature) - c['F']))

    def inverse(self, signal):
        c = self.c
        return c['B'] / torch.log(c['R1'] / (c['R2'] * signal.clamp_min(1e-8)) + c['F'])


def frozen_matrix(document, native_size):
    if document.get('kind') != 'approximate_FLIR_display_adapter':
        raise ValueError('Expected existing fit_flir_display_mapping.py mapping.json')
    p = np.asarray(document['parameters']['geometry'], dtype=float)
    if p.shape != (6,) or not np.isfinite(p).all():
        raise ValueError('Invalid frozen affine')
    # This historical display adapter was defined at the released 640x480 grid.
    W, H = 640., 480.
    w, h = native_size
    def at(X, Y):
        x, y = 2*(X+.5)/W-1, 2*(Y+.5)/H-1
        gx = (1+p[0]/W)*x + p[1]/W*y + 2*p[2]/W
        gy = p[3]/H*x + (1+p[4]/H)*y + 2*p[5]/H
        return np.array([(gx+1)*w/2-.5, (gy+1)*h/2-.5])
    b = at(0, 0)
    A = np.eye(3)
    A[:2] = np.column_stack((at(1,0)-b, at(0,1)-b, b))
    if abs(np.linalg.det(A)) < 1e-8:
        raise ValueError('Singular affine')
    return A


def inverse_grid(matrix, native_shape, device):
    h, w = native_shape
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    xy = np.stack((xx, yy, np.ones_like(xx)), -1) @ np.linalg.inv(matrix).T
    # grid_sample align_corners=False, in the existing 640x480 thermal frame.
    grid = np.stack((2*(xy[...,0]+.5)/640.-1, 2*(xy[...,1]+.5)/480.-1), -1)
    valid = (xy[...,0] >= -.5) & (xy[...,0] <= 639.5) & (xy[...,1] >= -.5) & (xy[...,1] <= 479.5)
    return torch.tensor(grid[None], dtype=torch.float32, device=device), torch.tensor(valid, device=device)


def evenly(items, limit):
    if not limit or limit >= len(items):
        return list(items)
    return [items[int(i)] for i in np.linspace(0, len(items)-1, limit, dtype=int)]


class ShuffledViews:
    """Visit every fit view once per shuffled pass; independent of global RNG."""
    def __init__(self, names, seed):
        self.names = list(names)
        if not self.names:
            raise ValueError('No fit views')
        self.generator = torch.Generator().manual_seed(seed)
        self.order = []
        self.views_seen = 0

    def take(self, count):
        names = []
        for _ in range(count):
            if not self.order:
                self.order = torch.randperm(len(self.names), generator=self.generator).tolist()
            names.append(self.names[self.order.pop()])
            self.views_seen += 1
        return names
