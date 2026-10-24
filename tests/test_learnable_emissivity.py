from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch

from test_linear_emissivity import options
from scene.gaussian_model import GaussianModel
from utils.emissivity_models import (learnable_emissivity_settings,
                                     endpoint_emissivity_parameters)
from utils.probe_maps import build_probe_fields
from utils.radiative_utils import (PlanckLUT, fit_material_decoupled_rt_attributes,
                                   _attach_material_fit_summary)


class LearnableEmissivityTests(unittest.TestCase):
    def config(self, **changes):
        return learnable_emissivity_settings(options(phys_probe_mode="learnable_linear_emissivity", **changes))

    def test_endpoint_bounds_cover_entire_interval_and_both_signs(self):
        cfg = self.config()
        center = torch.tensor([-20., -1., 0., 1., 20.])
        for difference in (-30., -2., 0., 2., 30.):
            eps0, slope, e_low, e_high = endpoint_emissivity_parameters(
                center, torch.full_like(center, difference), cfg)
            temperatures = torch.linspace(260., 360., 101)[:, None]
            e = eps0 + slope*(temperatures-300.)
            self.assertGreaterEqual(float(e.min()), 0.85-1e-6)
            self.assertLessEqual(float(e.max()), 0.99+1e-6)
            self.assertTrue(torch.allclose(e[0], e_low, atol=1e-7))
            self.assertTrue(torch.allclose(e[-1], e_high, atol=1e-7))
            if difference:
                self.assertTrue((slope*difference >= 0).all())

    def test_warmup_freezes_only_difference_and_unfreezes_from_zero(self):
        cfg = self.config()
        center = torch.tensor([0.3], requires_grad=True)
        difference = torch.zeros(1, requires_grad=True)
        eps0, slope, _, _ = endpoint_emissivity_parameters(center, difference, cfg, freeze_slope=True)
        (eps0 + 20.*slope).sum().backward()
        self.assertIsNone(difference.grad)
        self.assertGreater(float(center.grad.abs()), 0.)
        self.assertEqual(float(slope), 0.)
        center.grad = None
        eps0, slope, _, _ = endpoint_emissivity_parameters(center, difference, cfg)
        (eps0 + 20.*slope).sum().backward()
        self.assertTrue(torch.isfinite(difference.grad).all())
        self.assertGreater(float(difference.grad.abs()), 0.)

    def test_temperature_gradient_remains_connected(self):
        cfg = self.config()
        lut = PlanckLUT(260., 360., 7.5, 13.5, "cpu")
        raw = torch.tensor([-0.4, 0.6], requires_grad=True)
        center = torch.tensor([0.2], requires_grad=True)
        difference = torch.tensor([0.8], requires_grad=True)
        eps0, slope, _, _ = endpoint_emissivity_parameters(center, difference, cfg)
        e = eps0 + slope*(lut.norm_radiance_to_temp(torch.sigmoid(raw))-300.)
        e.sum().backward()
        for grad in (raw.grad, center.grad, difference.grad):
            self.assertTrue(torch.isfinite(grad).all())
            self.assertTrue((grad.abs() > 0).all())

    def test_known_temperature_synthetic_data_recovers_opposite_slopes(self):
        # This checks the identifiable subproblem with independently known T;
        # it does not claim recovery in the jointly unknown-T image problem.
        cfg = self.config()
        temperature = torch.linspace(265., 355., 31)[:, None]
        lut = PlanckLUT(260., 360., 7.5, 13.5, "cpu")
        u = torch.tensor(np.interp(temperature.numpy(), lut.temps.numpy(), lut.radiance_norm.numpy()))
        target_eps0 = torch.tensor([0.94, 0.93])
        target_slope = torch.tensor([0.0004, -0.0003])
        q = 0.1 + (target_eps0 + target_slope*(temperature-300.))*(u-0.1)
        center = torch.zeros(2, requires_grad=True)
        difference = torch.zeros(2, requires_grad=True)
        optimizer = torch.optim.Adam([center, difference], lr=0.08)
        for _ in range(400):
            eps0, slope, _, _ = endpoint_emissivity_parameters(center, difference, cfg)
            prediction = 0.1 + (eps0+slope*(temperature-300.))*(u-0.1)
            loss = ((prediction-q)/0.14).square().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        self.assertTrue(torch.allclose(eps0, target_eps0, atol=2e-5, rtol=0))
        self.assertTrue(torch.allclose(slope, target_slope, atol=1e-6, rtol=0))

    def test_fit_cache_export_and_summary_use_actual_cluster_slopes(self):
        n = 24
        g = GaussianModel(0, "cpu")
        g._xyz = torch.arange(n*3, dtype=torch.float32).reshape(n, 3)/n
        q = torch.linspace(0.08, 0.92, n)[:, None]
        extracted = {"q_target": q, "rgb_dc": q.expand(-1, 3), "rest_ratio": torch.zeros_like(q)}
        opt = options(phys_probe_mode="learnable_linear_emissivity")
        lut = PlanckLUT(260., 360., 7.5, 13.5, "cpu")
        torch.manual_seed(123)
        fit = fit_material_decoupled_rt_attributes(extracted, g, opt, "cpu", lut)
        self.assertGreater(float(fit["cluster_k"].abs().max()), 1e-8)
        self.assertTrue(torch.equal(fit["emissivity_slope"], fit["cluster_k"][fit["material_ids"].reshape(-1)]))
        metadata = fit["probe_metadata"]
        self.assertIsNone(metadata["slope_per_K"])
        self.assertEqual(metadata["schema_version"], 3)
        state = dict(extracted, **fit)
        fields = build_probe_fields(state, n)
        self.assertIn("emissivity_slope", fields)
        summary = {}
        _attach_material_fit_summary(summary, fit)
        ends = torch.cat([fit["cluster_eps0"] + fit["cluster_k"]*(t-300.) for t in (260., 360.)])
        self.assertAlmostEqual(summary["emissivity_min_over_T_range"], float(ends.min()))
        self.assertAlmostEqual(summary["emissivity_max_over_T_range"], float(ends.max()))
        self.assertGreaterEqual(float(ends.min()), 0.85-1e-6)
        self.assertLessEqual(float(ends.max()), 0.99+1e-6)
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp)/"probe.pt")
            names = ("q_target", "q_phys", "temperature", "emissivity", "emission", "ambient", "delta",
                     "rest_ratio", "rgb_dc", "emissivity_ref", "emissivity_slope", "material_ids",
                     "cluster_eps0", "cluster_k", "cluster_counts", "probe_metadata")
            g.update_physical_probe(12, **{name: state[name] for name in names})
            self.assertTrue(g.save_physical_probe(path))
            restored = GaussianModel(0, "cpu")
            restored._xyz = g._xyz.clone()
            self.assertTrue(restored.load_physical_probe(path))
            self.assertTrue(torch.equal(restored._phys_cluster_k, fit["cluster_k"]))
            self.assertEqual(restored._phys_probe_metadata, metadata)
        state["emissivity_slope"] = state["emissivity_slope"] + 0.001
        with self.assertRaises(ValueError):
            build_probe_fields(state, n)

    def test_invalid_options_rejected_before_fit(self):
        for changes in ({"phys_probe_lambda_slope": -0.1},
                        {"phys_probe_slope_warmup_fraction": 1.},
                        {"phys_probe_slope_lr_scale": 0.},
                        {"phys_probe_slope_lr_scale": float("nan")},
                        {"phys_probe_emissivity_slope": 0.0001}):
            with self.assertRaises(ValueError):
                self.config(**changes)


if __name__ == "__main__":
    unittest.main()
