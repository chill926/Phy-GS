import unittest
import torch

from utils.probe_maps import build_probe_fields, decode_field, encode_field


class ProbeMapTests(unittest.TestCase):
    def test_signed_weighted_mean_is_not_background_biased(self):
        values = torch.tensor([-0.04, 0.02])
        weights = torch.tensor([0.12, 0.18])
        encoded, low, span = encode_field(values)
        image, valid = decode_field((encoded * weights).sum(), weights.sum(), low, span, 0.1)
        self.assertTrue(valid.item())
        self.assertAlmostEqual(image.item(), ((values * weights).sum() / weights.sum()).item(), places=7)
        invalid, mask = decode_field(torch.tensor(0.), torch.tensor(0.), low, span, 0.1)
        self.assertFalse(mask.item())
        self.assertTrue(torch.isnan(invalid).item())

    def test_constant_temperature_survives_partial_coverage(self):
        encoded, low, span = encode_field(torch.tensor([300., 300.]))
        image, _ = decode_field(encoded.sum() * 0.1, torch.tensor(0.2), low, span, 0.1)
        self.assertEqual(image.item(), 300.)

    def test_radiance_decomposition_and_point_count(self):
        state = {"temperature": torch.tensor([300., 310.]),
                 "emissivity": torch.tensor([0.9, 0.95]),
                 "emission": torch.tensor([0.4, 0.8]),
                 "ambient": torch.tensor([0.1, 0.1]),
                 "delta": torch.tensor([-0.01, 0.01]),
                 "q_target": torch.tensor([0.36, 0.78])}
        state["q_phys"] = state["emissivity"] * state["emission"] + (1-state["emissivity"]) * state["ambient"] + state["delta"]
        fields = build_probe_fields(state, 2)
        self.assertTrue(torch.allclose(fields["q_phys_proxy"], fields["self_emission_proxy"] + fields["reflected_ambient_proxy"] + fields["residual_proxy"]))
        with self.assertRaises(ValueError):
            build_probe_fields(state, 3)


if __name__ == "__main__":
    unittest.main()
