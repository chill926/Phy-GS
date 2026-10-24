from argparse import ArgumentParser
from types import SimpleNamespace
from pathlib import Path
import tempfile
import unittest
import torch

from arguments import OptimizationParams
from scene.gaussian_model import GaussianModel
from utils.emissivity_models import linear_emissivity_settings, linear_emissivity
from utils.radiative_utils import PlanckLUT, fit_material_decoupled_rt_attributes


def options(**changes):
    opt = OptimizationParams(ArgumentParser())
    opt.phys_decouple_clusters = 3
    opt.phys_decouple_cluster_samples = 24
    opt.phys_decouple_cluster_iters = 3
    opt.phys_decouple_smooth_samples = 12
    opt.phys_decouple_steps = 8
    for k, v in changes.items():
        setattr(opt, k, v)
    return opt


class LinearEmissivityTests(unittest.TestCase):
    def test_entire_interval_bounds_for_both_slope_signs(self):
        for slope in (-0.0004, 0.0004):
            cfg = linear_emissivity_settings(options(phys_probe_emissivity_slope=slope))
            eps0 = cfg["eps0_min"] + (cfg["eps0_max"]-cfg["eps0_min"]) * torch.sigmoid(torch.tensor([-20., 0., 20.]))
            for temperature in torch.linspace(260., 360., 101):
                e = linear_emissivity(eps0, temperature.expand(3), torch.arange(3), cfg)
                self.assertGreaterEqual(float(e.min()), 0.85-1e-6)
                self.assertLessEqual(float(e.max()), 0.99+1e-6)

    def test_temperature_coupling_has_finite_nonzero_gradient(self):
        cfg = linear_emissivity_settings(options(phys_probe_emissivity_slope=0.0004))
        lut = PlanckLUT(260., 360., 7.5, 13.5, "cpu")
        u_raw = torch.tensor([-0.4, 0.6], requires_grad=True)
        eps0_raw = torch.tensor([0.2], requires_grad=True)
        temperature = lut.norm_radiance_to_temp(torch.sigmoid(u_raw))
        eps0 = cfg["eps0_min"] + (cfg["eps0_max"]-cfg["eps0_min"]) * torch.sigmoid(eps0_raw)
        e = linear_emissivity(eps0, temperature, torch.zeros(2, dtype=torch.long), cfg)
        e.sum().backward()
        for value in (u_raw.grad, eps0_raw.grad):
            self.assertTrue(torch.isfinite(value).all())
            self.assertTrue((value.abs() > 0).all())

    def test_invalid_physical_settings_are_rejected(self):
        for changes in ({"phys_probe_emissivity_slope": 0.01},
                        {"phys_probe_temp_ref": 400.},
                        {"phys_probe_emissivity_slope": float("nan")},
                        {"phys_probe_e_init": 0.851, "phys_probe_emissivity_slope": 0.0004}):
            with self.assertRaises(ValueError):
                linear_emissivity_settings(options(**changes))

    def test_zero_slope_reproduces_original_and_nonzero_slope_changes_fit(self):
        n = 24
        xyz = torch.arange(n*3, dtype=torch.float32).reshape(n, 3) / n
        gaussians = SimpleNamespace(get_xyz=xyz)
        q = torch.linspace(0.08, 0.92, n)[:, None]
        extracted = {"q_target": q, "rgb_dc": q.expand(-1, 3), "rest_ratio": torch.zeros_like(q)}
        lut = PlanckLUT(260., 360., 7.5, 13.5, "cpu")
        results = []
        for mode, slope in (("original", 0.), ("linear_emissivity", 0.), ("linear_emissivity", 0.0004)):
            opt = options(phys_probe_mode=mode, phys_probe_emissivity_slope=slope)
            torch.manual_seed(123)
            results.append(fit_material_decoupled_rt_attributes(extracted, gaussians, opt, "cpu", lut))
        for name in ("temperature", "emissivity", "q_phys", "delta"):
            self.assertTrue(torch.allclose(results[0][name], results[1][name], atol=1e-6, rtol=1e-6), name)
        fit = results[2]
        self.assertFalse(torch.allclose(fit["temperature"], results[0]["temperature"]))
        expected = fit["emissivity_ref"] + fit["emissivity_slope"] * (fit["temperature"]-300.)
        self.assertTrue(torch.allclose(expected, fit["emissivity"]))
        self.assertGreaterEqual(float(fit["emissivity"].min()), 0.85)
        self.assertLessEqual(float(fit["emissivity"].max()), 0.99)

    def test_cache_roundtrip_old_cache_and_clear(self):
        g = GaussianModel(0, "cpu")
        g._xyz = torch.zeros(2, 3)
        state = {name: torch.ones(2, 1) for name in
                 ("q_target", "q_phys", "temperature", "emissivity", "emission", "ambient", "delta", "rest_ratio")}
        state.update(rgb_dc=torch.ones(2, 3), emissivity_ref=torch.full((2, 1), 0.95),
                     emissivity_slope=torch.full((2, 1), 0.0001),
                     material_ids=torch.zeros(2, 1, dtype=torch.long),
                     cluster_eps0=torch.tensor([[0.95]]), cluster_k=torch.tensor([[0.0001]]),
                     probe_metadata={"mode": "linear_emissivity", "temperature_ref_K": 300.})
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp)/"probe.pt")
            g.update_physical_probe(12, **state)
            self.assertTrue(g.save_physical_probe(path))
            h = GaussianModel(0, "cpu")
            h._xyz = torch.zeros(2, 3)
            self.assertTrue(h.load_physical_probe(path))
            self.assertEqual(h._phys_probe_metadata, state["probe_metadata"])
            self.assertTrue(torch.equal(h._phys_emissivity_slope, state["emissivity_slope"]))
            old = {k: v for k, v in torch.load(path).items()
                   if k not in g._probe_optional_tensor_fields and k != "probe_metadata"}
            torch.save(old, path)
            self.assertTrue(h.load_physical_probe(path))
            self.assertEqual(h._phys_probe_metadata, {})
            self.assertEqual(h._phys_emissivity_slope.numel(), 0)
            h.clear_physical_probe()
            self.assertFalse(h.has_physical_probe())


if __name__ == "__main__":
    unittest.main()
