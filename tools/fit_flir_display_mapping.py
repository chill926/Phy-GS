#!/usr/bin/env python3
"""Fit a shared FLIR DN -> released pseudocolor display approximation.

Independent of PhysIR-Splat training. Dependencies: numpy, Pillow, torch.
The FFF record layout follows the public ExifTool FLIR format description:
https://github.com/exiftool/exiftool/blob/master/lib/Image/ExifTool/FLIR.pm
This is an approximate display adapter, NOT recovered original FLIR software,
temperature calibration, or a temperature ground-truth generator.

fit: only thermal/train supplies fitting and validation frames. Whole frames
are held out; thermal/test is never loaded by fit. All frames share parameters.
evaluate/apply: frozen checkpoint, no reference-dependent alignment or fitting.
"""
import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import struct
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def extract_flir(path, png_byte_order='swap'):
    """Read JPEG APP1 FLIR/FFF records; fail explicitly on unsupported input.

    swap is explicit for the audited RoadBlock PNGs, not an intensity heuristic.
    native supports cameras whose embedded PNG has correct sample byte order.
    """
    blob = Path(path).read_bytes()
    if not blob.startswith(b'\xff\xd8'):
        raise ValueError('Expected a radiometric FLIR JPEG: ' + str(path))
    pos, parts = 2, []
    while pos < len(blob):
        if blob[pos] != 255:
            raise ValueError('Invalid JPEG marker')
        while blob[pos] == 255:
            pos += 1
        marker = blob[pos]
        pos += 1
        if marker in (0xda, 0xd9):
            break
        if marker == 1 or 0xd0 <= marker <= 0xd7:
            continue
        length = int.from_bytes(blob[pos:pos+2], 'big')
        if length < 2 or pos + length > len(blob):
            raise ValueError('Truncated JPEG segment')
        payload = blob[pos+2:pos+length]
        pos += length
        if marker == 0xe1 and payload.startswith(b'FLIR\x00'):
            if len(payload) < 8:
                raise ValueError('Truncated FLIR chunk')
            parts.append((payload[6], payload[7], payload[8:]))
    if not parts:
        raise ValueError('No FLIR radiometric records: ' + str(path))
    totals = {p[1] for p in parts}
    if len(totals) != 1 or [p[0] for p in parts] != list(range(len(parts))) or parts[0][1] != len(parts)-1:
        raise ValueError('Unsupported or incomplete FLIR chunk sequence')
    fff = b''.join(p[2] for p in parts)
    if fff[:4] not in (b'FFF\x00', b'AFF\x00'):
        raise ValueError('Missing FLIR FFF header')
    endian = next((e for e in ('<', '>') if 100 <= struct.unpack_from(e+'I', fff, 20)[0] < 200), None)
    if endian is None:
        raise ValueError('Unsupported FFF version')
    directory, count = struct.unpack_from(endian+'II', fff, 24)
    if directory + count*32 > len(fff):
        raise ValueError('Truncated FFF directory')
    records = {}
    for j in range(count):
        entry = directory + j*32
        kind = struct.unpack_from(endian+'H', fff, entry)[0]
        start, size = struct.unpack_from(endian+'II', fff, entry+12)
        if start + size > len(fff):
            raise ValueError('Truncated FFF record')
        if kind in (1, 0x20, 0x22):
            if kind in records:
                raise ValueError('Multiple records of required type')
            records[kind] = fff[start:start+size]
    raw, camera, palette_record = (records[k] for k in (1, 0x20, 0x22))
    raw_endian = '<' if raw[:2] == b'\x02\x00' else '>'
    width, height = struct.unpack_from(raw_endian+'HH', raw, 2)
    if not raw[32:].startswith(b'\x89PNG\r\n\x1a\n'):
        raise ValueError('This reader only supports embedded 16-bit PNG')
    dn = np.asarray(Image.open(io.BytesIO(raw[32:])))
    if dn.ndim != 2 or dn.shape != (height, width) or dn.min() < 0 or dn.max() > 65535:
        raise ValueError('Unsupported thermal PNG shape/range')
    dn = dn.astype(np.uint16)
    if png_byte_order == 'swap':
        dn = dn.byteswap()
    camera_endian = '<' if camera[:2] == b'\x02\x00' else '>'
    median = struct.unpack_from(camera_endian+'H', camera, 0x338)[0]
    span = struct.unpack_from(camera_endian+'H', camera, 0x33c)[0]
    if span <= 0:
        raise ValueError('Nonpositive display range')
    colors = palette_record[0]
    palette = np.frombuffer(palette_record[0x70:0x70+3*colors], np.uint8).reshape(colors, 3).copy()
    meta = {'median': median, 'span': span, 'shape': list(dn.shape),
            'png_byte_order': png_byte_order, 'dn_min': int(dn.min()), 'dn_max': int(dn.max()),
            'palette_sha256': hashlib.sha256(palette.tobytes()).hexdigest()}
    meta['planck'] = {name: struct.unpack_from(camera_endian+fmt, camera, offset)[0]
                      for name, fmt, offset in [('R1','f',0x58), ('B','f',0x5c),
                                               ('F','f',0x60), ('O','i',0x308), ('R2','f',0x30c)]}
    return dn, palette, meta


def palette_rgb(ycc):
    # Reference BT.601 limited-range conversion; exact Studio conversion unknown.
    y = ycc[:, 0].astype(np.float64)-16
    cr = ycc[:, 1].astype(np.float64)-128
    cb = ycc[:, 2].astype(np.float64)-128
    return np.clip(np.stack([1.16438356*y+1.59602678*cr,
                             1.16438356*y-.81296765*cr-.39176229*cb,
                             1.16438356*y+2.01723214*cb], -1), 0, 255)/255


class DisplayMapping(torch.nn.Module):
    def __init__(self, palette, model='linear', order='resize_then_colorize', knots=12):
        super().__init__()
        self.model, self.order, self.knots = model, order, knots
        self.register_buffer('palette', torch.tensor(palette, dtype=torch.float32))
        # Output-normalized coordinates -> source-normalized coordinates.
        # All six coefficients use target-pixel units for optimizer conditioning.
        self.geometry = torch.nn.Parameter(torch.zeros(6))
        self.log_gain = torch.nn.Parameter(torch.zeros(()))
        self.offset = torch.nn.Parameter(torch.zeros(()))
        self.curve_logits = torch.nn.Parameter(torch.zeros(knots-1), requires_grad=model == 'monotone')

    def curve(self, value):
        value = (self.log_gain.exp()*value + self.offset).clamp(0, 1)
        if self.model == 'monotone':
            heights = torch.cat([value.new_zeros(1), self.curve_logits.softmax(0).cumsum(0)])
            s = value*(self.knots-1)
            lo = s.floor().long().clamp(0, self.knots-2)
            value = heights[lo] + (s-lo)*(heights[lo+1]-heights[lo])
        index = value*(len(self.palette)-1)
        lo = index.floor().long().clamp(0, len(self.palette)-2)
        fraction = (index-lo).unsqueeze(-1)
        return self.palette[lo]*(1-fraction) + self.palette[lo+1]*fraction

    def forward(self, dn_normalized, xy, target_size):
        """dn_normalized [B,H,W], xy [B,N,2] in output normalized coordinates."""
        height, width = target_size
        p = self.geometry
        gx = (1+p[0]/width)*xy[..., 0] + p[1]/width*xy[..., 1] + 2*p[2]/width
        gy = p[3]/height*xy[..., 0] + (1+p[4]/height)*xy[..., 1] + 2*p[5]/height
        h, w = dn_normalized.shape[-2:]
        x, y = ((gx+1)*w/2-.5).clamp(0, w-1), ((gy+1)*h/2-.5).clamp(0, h-1)
        x0, y0 = x.floor().long(), y.floor().long()
        x1, y1 = (x0+1).clamp(max=w-1), (y0+1).clamp(max=h-1)
        flat = dn_normalized.flatten(1)
        values = [flat.gather(1, yy*w+xx) for xx, yy in ((x0,y0),(x1,y0),(x0,y1),(x1,y1))]
        wx, wy = x-x0, y-y0
        weights = [(1-wx)*(1-wy), wx*(1-wy), (1-wx)*wy, wx*wy]
        if self.order == 'resize_then_colorize':
            return self.curve(sum(v*t for v, t in zip(values, weights)))
        return sum(self.curve(v)*t.unsqueeze(-1) for v, t in zip(values, weights))

    def regularization(self):
        # Weak preference for a smooth monotone display curve and small geometry.
        reg = 1e-7*self.geometry.square().mean()
        if self.model == 'monotone':
            slopes = self.curve_logits.softmax(0)*(self.knots-1)
            reg = reg + 1e-5*(slopes[1:]-slopes[:-1]).square().mean()
        return reg

    def constrain(self):
        with torch.no_grad():
            self.geometry.clamp_(-12, 12)
            self.log_gain.clamp_(-.5, .5)
            self.offset.clamp_(-.3, .3)
            self.curve_logits.clamp_(-3, 3)


def read_frames(scene, split, byte_order, names=None):
    paths = sorted((Path(scene)/'thermal'/split).glob('*.jpg'))
    if names is not None:
        paths = [p for p in paths if p.stem in set(names)]
    if not paths:
        raise ValueError('No paired frames: ' + str(Path(scene)/'thermal'/split))
    frames, reference_palette = [], None
    for target_path in paths:
        raw_path = Path(scene)/'raw_images'/target_path.name
        dn, palette, meta = extract_flir(raw_path, byte_order)
        if reference_palette is None:
            reference_palette = palette
        if not np.array_equal(palette, reference_palette):
            raise ValueError('Palettes differ; one shared palette is not applicable')
        target = np.asarray(Image.open(target_path).convert('RGB')).copy()
        frames.append({'name': target_path.stem, 'dn': dn, 'target': target, 'meta': meta,
                       'raw_path': str(raw_path.resolve()), 'target_path': str(target_path.resolve()),
                       'raw_sha256': sha256(raw_path), 'target_sha256': sha256(target_path)})
    return frames, reference_palette


def normalized_dn(frame, device):
    return torch.tensor((frame['dn'].astype(np.float32)-frame['meta']['median'])/frame['meta']['span']+.5,
                        dtype=torch.float32, device=device)


def sample_frames(frames, pixels, device, seed):
    rng = np.random.RandomState(seed)
    dns, xys, rgbs = [], [], []
    shape = frames[0]['target'].shape[:2]
    for frame in frames:
        if frame['target'].shape[:2] != shape or frame['dn'].shape != frames[0]['dn'].shape:
            raise ValueError('Mixed image sizes unsupported in a single fit')
        h, w = shape
        indices = rng.choice(h*w, size=min(pixels, h*w), replace=False)
        yy, xx = indices//w, indices % w
        xy = np.stack([2*(xx+.5)/w-1, 2*(yy+.5)/h-1], -1)
        dns.append(normalized_dn(frame, device))
        xys.append(torch.tensor(xy, dtype=torch.float32, device=device))
        rgbs.append(torch.tensor(frame['target'].reshape(-1, 3)[indices]/255., dtype=torch.float32, device=device))
    return torch.stack(dns), torch.stack(xys), torch.stack(rgbs), shape


@torch.no_grad()
def render_frame(model, frame, device, chunk=32768):
    h, w = frame['target'].shape[:2]
    dn = normalized_dn(frame, device).unsqueeze(0)
    output = []
    for start in range(0, h*w, chunk):
        idx = torch.arange(start, min(start+chunk, h*w), device=device)
        xy = torch.stack([2*((idx % w)+.5)/w-1, 2*(torch.div(idx, w, rounding_mode='floor')+.5)/h-1], -1)
        rgb = model(dn, xy.unsqueeze(0), (h, w))[0]
        output.append((rgb.clamp(0, 1)*255).round().to(torch.uint8).cpu().numpy())
    return np.concatenate(output).reshape(h, w, 3)


def metrics(pred, target, device):
    p = torch.tensor(pred.transpose(2,0,1).copy(), dtype=torch.float32, device=device).unsqueeze(0)/255
    t = torch.tensor(target.transpose(2,0,1).copy(), dtype=torch.float32, device=device).unsqueeze(0)/255
    mse = (p-t).square().mean().item()
    g = torch.exp(-((torch.arange(11, device=device)-5).float().square())/(2*1.5**2))
    g = g/g.sum()
    window = (g[:,None]*g[None,:]).expand(3,1,11,11).contiguous()
    mu1, mu2 = F.conv2d(p,window,padding=5,groups=3), F.conv2d(t,window,padding=5,groups=3)
    s1 = F.conv2d(p*p,window,padding=5,groups=3)-mu1*mu1
    s2 = F.conv2d(t*t,window,padding=5,groups=3)-mu2*mu2
    cov = F.conv2d(p*t,window,padding=5,groups=3)-mu1*mu2
    ss = (((2*mu1*mu2+.01**2)*(2*cov+.03**2))/
          ((mu1.square()+mu2.square()+.01**2)*(s1+s2+.03**2))).mean().item()
    return {'PSNR': 10*math.log10(1/max(mse, 1e-15)), 'SSIM': ss,
            'MAE_8bit': float(np.abs(pred.astype(float)-target.astype(float)).mean()),
            'max_abs_error_8bit': int(np.abs(pred.astype(int)-target.astype(int)).max())}


def evaluate_frames(model, frames, output, device):
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    with torch.no_grad():
        for frame in frames:
            pred = render_frame(model, frame, device)
            row = {'image': frame['name'], **metrics(pred, frame['target'], device)}
            rows.append(row)
            Image.fromarray(pred).save(output/(frame['name']+'_mapped.png'))
            error = np.clip(np.abs(pred.astype(float)-frame['target'])*6, 0, 255).astype(np.uint8)
            panel = np.concatenate([frame['target'], pred, error], axis=1)
            Image.fromarray(panel).save(output/(frame['name']+'_comparison.png'))
            print(json.dumps(row), flush=True)
    mean = {key: float(np.mean([r[key] for r in rows])) for key in ('PSNR','SSIM','MAE_8bit')}
    result = {'mean': mean, 'per_frame': rows, 'frame_count': len(rows),
              'protocol': '8-bit rounded RGB pseudocolor; PSNR and SSIM (11x11 Gaussian sigma=1.5, zero padding), arithmetic frame mean',
              'reference_dependent_refitting': False, 'original_mapping_confirmed': False}
    write_json(output/'metrics.json', result)
    return result


def save_model(path, model, palette_ycc, args, train, validation, history):
    params = {k: v.detach().cpu().numpy().tolist() for k,v in model.named_parameters()}
    value = {'schema_version': 1, 'kind': 'approximate_FLIR_display_adapter',
             'original_mapping_confirmed': False, 'temperature_calibration': False,
             'model': model.model, 'order': model.order, 'knots': model.knots,
             'parameters': params, 'palette_rgb_01': model.palette.cpu().numpy().tolist(),
             'palette_ycrcb': palette_ycc.tolist(), 'png_byte_order': args.png_byte_order,
             'normalization': 'u=(DN-RawValueMedian)/RawValueRange+0.5; shared gain and offset follow',
             'geometry': 'gx=(1+p0/W)*x+(p1/W)*y+2*p2/W; gy=(p3/H)*x+(1+p4/H)*y+2*p5/H; x,y normalized; shared inverse affine; border replicate; bilinear sampling',
             'fit_frames': [f['name'] for f in train], 'validation_frames': [f['name'] for f in validation],
             'test_used_for_fitting_or_selection': False, 'selection': 'lowest fixed validation sampled RGB MSE',
             'settings': vars(args), 'history': history,
             'script_sha256': sha256(__file__)}
    write_json(path, value)


def load_model(path, device):
    data = json.loads(Path(path).read_text())
    model = DisplayMapping(data['palette_rgb_01'], data['model'], data['order'], data['knots']).to(device)
    with torch.no_grad():
        for key, value in model.named_parameters():
            value.copy_(torch.tensor(data['parameters'][key], device=device))
    model.eval()
    return model, data


def fit(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    frames, palette = read_frames(args.scene, 'train', args.png_byte_order)
    validation = [f for j,f in enumerate(frames) if j % args.val_every == 0]
    train = [f for j,f in enumerate(frames) if j % args.val_every != 0]
    if len(train) < 2 or len(validation) < 2:
        raise ValueError('Need at least two fit and two validation frames')
    manifest = [{k:f[k] for k in ('name','raw_path','target_path','raw_sha256','target_sha256','meta')}
                for f in frames]
    write_json(output/'manifest.json', manifest)
    train_data = sample_frames(train, args.pixels, device, args.seed)
    val_data = sample_frames(validation, args.pixels, device, args.seed+1)
    model = DisplayMapping(palette_rgb(palette), args.model, args.order, args.knots).to(device)
    if args.geometry == 'identity':
        model.geometry.requires_grad_(False)
    optimizer = torch.optim.Adam([
        {'params': [model.geometry], 'lr': .015},
        {'params': [model.log_gain, model.offset, model.curve_logits], 'lr': .002}])
    best, history, started = float('inf'), [], time.time()
    print('fit={} validation={} test=0 device={} model={} order={}'.format(
        len(train), len(validation), device, args.model, args.order), flush=True)
    for step in range(args.steps+1):
        if step:
            optimizer.zero_grad()
            loss = (model(train_data[0],train_data[1],train_data[3])-train_data[2]).square().mean()
            (loss+model.regularization()).backward()
            optimizer.step()
            model.constrain()
        if step % args.log_every == 0 or step == args.steps:
            with torch.no_grad():
                train_mse = (model(train_data[0],train_data[1],train_data[3])-train_data[2]).square().mean().item()
                val_mse = (model(val_data[0],val_data[1],val_data[3])-val_data[2]).square().mean().item()
            row = {'step':step, 'train_PSNR_sampled':-10*math.log10(max(train_mse,1e-15)),
                   'val_PSNR_sampled':-10*math.log10(max(val_mse,1e-15)), 'seconds':round(time.time()-started,2)}
            history.append(row)
            print(json.dumps(row), flush=True)
            if val_mse < best:
                best = val_mse
                save_model(output/'mapping.json', model, palette, args, train, validation, history)
    write_json(output/'history.json', history)
    model, _ = load_model(output/'mapping.json', device)
    result = evaluate_frames(model, validation, output/'validation', device)
    print('VALIDATION_MEAN='+json.dumps(result['mean']), flush=True)
    print('CHECKPOINT='+str((output/'mapping.json').resolve()), flush=True)


def evaluate(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    model, data = load_model(args.checkpoint, device)
    names = data['validation_frames'] if args.split == 'validation' else None
    frames, palette = read_frames(args.scene, 'train' if args.split == 'validation' else args.split,
                                  data['png_byte_order'], names)
    if not np.array_equal(palette, np.asarray(data['palette_ycrcb'], np.uint8)):
        raise ValueError('Checkpoint palette differs from evaluation palette')
    write_json(output/'evaluation_provenance.json', {'checkpoint':str(Path(args.checkpoint).resolve()),
               'checkpoint_sha256':sha256(args.checkpoint), 'split':args.split,
               'frames':[{k:f[k] for k in ('name','raw_sha256','target_sha256')} for f in frames]})
    result = evaluate_frames(model, frames, output, device)
    print('EVALUATION_MEAN='+json.dumps(result['mean']), flush=True)


def apply(args):
    device = torch.device(args.device)
    model, data = load_model(args.checkpoint, device)
    if args.raw:
        dn, palette, meta = extract_flir(args.raw, data['png_byte_order'])
        if not np.array_equal(palette, np.asarray(data['palette_ycrcb'], np.uint8)):
            raise ValueError('Different palette')
    else:
        if args.median is None or args.span is None or args.span <= 0:
            raise ValueError('--dn requires --median and positive --span in raw DN units')
        dn = np.load(args.dn, allow_pickle=False)
        if dn.ndim != 2 or not np.isfinite(dn).all():
            raise ValueError('Expected finite 2-D DN array')
        meta = {'median':args.median, 'span':args.span}
    frame = {'dn':dn, 'meta':meta, 'target':np.zeros((args.height,args.width,3),np.uint8)}
    path = Path(args.output)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(render_frame(model,frame,device)).save(path)
    print(str(path.resolve()))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('fit')
    p.add_argument('--scene', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--model', choices=['linear','monotone'], default='linear')
    p.add_argument('--order', choices=['resize_then_colorize','colorize_then_resize'], default='resize_then_colorize')
    p.add_argument('--geometry', choices=['affine','identity'], default='affine')
    p.add_argument('--knots', type=int, default=12)
    p.add_argument('--steps', type=int, default=800)
    p.add_argument('--pixels', type=int, default=2048, help='Fixed sampled pixels per whole frame for fitting/selection')
    p.add_argument('--val-every', type=int, default=5)
    p.add_argument('--log-every', type=int, default=50)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--png-byte-order', choices=['swap','native'], default='swap')
    p = sub.add_parser('evaluate')
    p.add_argument('--scene', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--split', choices=['train','validation','test'], default='validation')
    p.add_argument('--output', required=True)
    p = sub.add_parser('apply')
    p.add_argument('--checkpoint', required=True)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--raw')
    source.add_argument('--dn', help='2-D .npy array in raw DN units, already in native thermal-grid coordinates')
    p.add_argument('--median', type=float)
    p.add_argument('--span', type=float)
    p.add_argument('--height', type=int, default=480)
    p.add_argument('--width', type=int, default=640)
    p.add_argument('--output', required=True)
    for p in sub.choices.values():
        p.add_argument('--device', default='cpu')
        p.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    torch.set_num_threads(args.threads)
    if args.command == 'fit':
        if args.steps < 1 or args.pixels < 1 or args.val_every < 2 or args.log_every < 1 or args.knots < 3:
            parser.error('Invalid fitting counts')
        torch.manual_seed(args.seed)
    {'fit':fit,'evaluate':evaluate,'apply':apply}[args.command](args)


if __name__ == '__main__':
    main()
