"""Run from repo root: python -m unittest discover -s tests -v."""
from argparse import ArgumentParser
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch import nn
from arguments import OptimizationParams
from scene.gaussian_model import GaussianModel
from utils.general_utils import inverse_sigmoid
from utils.graphics_utils import BasicPointCloud


class IROpacityOffsetTests(unittest.TestCase):
    def model(self, enabled=True, adam_state=True):
        device = torch.device('cuda:0')
        model = GaussianModel(3, device)
        points = np.array([[x, y, z] for x in (0., 1.) for y in (0., 1.) for z in (2., 3.)], dtype=np.float32)
        pcd = BasicPointCloud(points=points, colors=np.full_like(points, .5), normals=np.zeros_like(points))
        model.create_from_pcd(pcd, 1., device, enable_thermal_branch=True)
        parser = ArgumentParser()
        opt = OptimizationParams(parser).extract(parser.parse_args([]))
        opt.ir_opacity_offset = enabled
        model.training_setup(opt, device)
        if adam_state:
            sum(group['params'][0].sum() * 0 for group in model.optimizer.param_groups).backward()
            model.optimizer.step()
            model.optimizer.zero_grad(set_to_none=True)
        return model

    def test_zero_identity_and_modality_gradient_routing(self):
        model = self.model()
        self.assertTrue(torch.equal(model.get_opacity, model.get_opacity_for_feature('thermal')))
        model.get_opacity_for_feature('rgb').sum().backward()
        self.assertIsNone(model._ir_opacity_offset.grad)
        model.optimizer.zero_grad(set_to_none=True)
        model.get_opacity_for_feature('thermal').sum().backward()
        self.assertTrue(torch.isfinite(model._ir_opacity_offset.grad).all())
        self.assertTrue((model._ir_opacity_offset.grad != 0).all())
        original_rgb = model.get_opacity.detach().clone()
        with torch.no_grad():
            model._ir_opacity_offset.fill_(2.)
        self.assertTrue(torch.equal(original_rgb, model.get_opacity_for_feature('rgb')))
        self.assertTrue((model.get_opacity_for_feature('ir') > original_rgb).all())

    def test_clone_split_prune_and_adam_follow_parent(self):
        model = self.model()
        with torch.no_grad():
            model._ir_opacity_offset.copy_(torch.arange(8, device='cuda').view(-1, 1) / 10)
            model._scaling.fill_(-8.)
        expected = model._ir_opacity_offset.detach().clone()
        grads = torch.ones((8, 1), device='cuda')
        model.densify_and_clone(grads, .5, 1., torch.device('cuda'))
        self.assertTrue(torch.equal(model._ir_opacity_offset, torch.cat([expected, expected])))
        self.assertEqual(model.optimizer.state[model._ir_opacity_offset]['exp_avg'].shape, (16, 1))
        remove = torch.zeros(16, dtype=torch.bool, device='cuda')
        remove[8:] = True
        model.prune_points(remove)
        with torch.no_grad():
            model._scaling.fill_(-1.)
        model.densify_and_split(grads, .5, 1., torch.device('cuda'), N=2)
        self.assertTrue(torch.equal(model._ir_opacity_offset, expected.repeat(2, 1)))
        self.assertEqual(model.optimizer.state[model._ir_opacity_offset]['exp_avg'].shape, (16, 1))
        model.get_opacity_for_feature('thermal').sum().backward()
        model.optimizer.step()
        self.assertTrue(torch.isfinite(model._ir_opacity_offset).all())

    def test_reset_caps_both_modalities_and_resets_moments(self):
        for adam_state in [False, True]:
            model = self.model(adam_state=adam_state)
            with torch.no_grad():
                model._opacity.copy_(inverse_sigmoid(torch.tensor([.001, .3] * 4, device='cuda').view(-1, 1)))
                model._ir_opacity_offset.copy_(torch.tensor([3., -7.] * 4, device='cuda').view(-1, 1))
            rgb = model.get_opacity.detach().clamp_max(.01)
            ir = model.get_opacity_for_feature('thermal').detach().clamp_max(.01)
            model.reset_opacity()
            torch.testing.assert_close(model.get_opacity, rgb)
            torch.testing.assert_close(model.get_opacity_for_feature('thermal'), ir)
            if adam_state:
                self.assertEqual(model.optimizer.state[model._ir_opacity_offset]['exp_avg'].count_nonzero().item(), 0)

    def test_pruning_retains_points_visible_in_either_modality(self):
        model = self.model()
        with torch.no_grad():
            model._opacity.fill_(-10.)
            model._ir_opacity_offset.fill_(12.)
        model.densify_and_prune(1., .005, 100., None, torch.device('cuda'))
        self.assertEqual(model.get_xyz.shape[0], 8)
        with torch.no_grad():
            model._ir_opacity_offset.fill_(0.)
        self.assertTrue((model.get_pruning_opacity < .005).all())

    def test_ply_roundtrip_legacy_and_resume_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = self.model()
            with torch.no_grad():
                model._ir_opacity_offset.copy_(torch.linspace(-1, 1, 8, device='cuda').view(-1, 1))
            path = str(Path(tmp) / 'offset.ply')
            model.save_ply(path)
            loaded = GaussianModel(3, torch.device('cuda'))
            loaded.load_ply(path, torch.device('cuda'))
            self.assertTrue(torch.equal(model._ir_opacity_offset, loaded._ir_opacity_offset))
            self.assertTrue(torch.equal(model.get_opacity_for_feature('thermal'), loaded.get_opacity_for_feature('thermal')))
            parser = ArgumentParser()
            opt = OptimizationParams(parser).extract(parser.parse_args([]))
            with self.assertRaisesRegex(ValueError, 'resume with'):
                loaded.training_setup(opt, torch.device('cuda'))
            opt.ir_opacity_offset = True
            loaded.training_setup(opt, torch.device('cuda'))
            legacy = self.model(enabled=False)
            legacy.save_ply(str(Path(tmp) / 'legacy.ply'))
            loaded.load_ply(str(Path(tmp) / 'legacy.ply'), torch.device('cuda'))
            self.assertFalse(loaded.has_ir_opacity_offset)
            self.assertTrue(torch.equal(loaded.get_opacity, loaded.get_opacity_for_feature('thermal')))


if __name__ == '__main__':
    unittest.main()
