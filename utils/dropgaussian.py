"""Progressive opacity dropout, adapted from DCVL-3D/DropGaussian_release.

The official 10k-step schedule ends at p=0.2. Normalize by this run's
iteration budget instead, so a 30k-step run also ends at p=0.2.
"""
import math

import torch


class ProgressiveIRDropGaussian:
    def __init__(self, total_iterations, max_rate=0.2, seed=0):
        if int(total_iterations) != total_iterations or total_iterations <= 0:
            raise ValueError("DropGaussian requires a positive integer iteration budget")
        if not math.isfinite(max_rate) or not 0.0 <= max_rate < 1.0:
            raise ValueError("ir_dropgaussian_max_rate must be finite and in [0, 1)")
        self.total_iterations = int(total_iterations)
        self.max_rate = float(max_rate)
        self.seed = int(seed)
        self._generators = {}

    def rate(self, iteration):
        return self.max_rate * min(1.0, max(0.0, iteration / self.total_iterations))

    def sample(self, opacity, iteration):
        """Return an N x 1 inverted-dropout multiplier and scheduled probability.

        A dedicated, step-seeded generator leaves global CPU/CUDA RNG untouched.
        The same step and point count reproduce the mask after a weights-only
        resume; this does not make the rest of training exactly resumable.
        No Gaussian parameter is changed here.
        """
        if opacity.ndim != 2 or opacity.shape[1] != 1:
            raise ValueError("DropGaussian expects N x 1 opacity")
        rate = self.rate(iteration)
        if rate == 0.0:
            return None, rate
        generator = self._generators.get(opacity.device)
        if generator is None:
            generator = torch.Generator(device=opacity.device)
            self._generators[opacity.device] = generator
        generator.manual_seed((self.seed + int(iteration)) % (2**63 - 1))
        keep = torch.rand(opacity.shape, device=opacity.device, dtype=opacity.dtype,
                          generator=generator) >= rate
        return keep.to(opacity.dtype) / (1.0 - rate), rate
