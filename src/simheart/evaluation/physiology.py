"""Physiology and smoothness scores of beats, with real-beat references."""

from collections.abc import Callable

import torch

from simheart.data.scaling import unit_to_sim
from simheart.evaluation.metrics import high_frequency_fraction
from simheart.losses.euler import best_euler_distance
from simheart.simulator.ecgsyn import EcgSyn
from simheart.simulator.fitting import ParamKDE

METRIC_K = 32  # parameter sets tried per beat for the physiology score
METRIC_SEED = 123  # same parameter draws at every evaluation, for every run


def euler_score(beats_unit: torch.Tensor, sampler: ParamKDE, sim: EcgSyn) -> float:
    """Median over beats of the smallest Euler distance across 32 sampled parameter sets.

    How close beats (on the [-1, 1] scale) are to *some* plausible simulator solution.
    The parameter draws are fixed, so every run is scored against the same draws.
    """
    generator = torch.Generator(device=beats_unit.device).manual_seed(METRIC_SEED)
    beats = unit_to_sim(beats_unit)
    return float(best_euler_distance(beats, sampler, sim, METRIC_K, generator).median())


def physiology_metrics(
    sim: EcgSyn, sampler: ParamKDE, real_val: torch.Tensor
) -> Callable[[torch.Tensor], dict[str, float]]:
    """A scorer for generated beats ([-1, 1] scale) that also reports real validation values.

    * euler_best_median: ``euler_score`` (lower is closer to the heart equations).
    * hf_fraction_median: median share of signal power above 40 Hz (jaggedness).
    """
    real_euler = euler_score(real_val, sampler, sim)
    real_hf = float(high_frequency_fraction(real_val).median())

    def metrics(generated: torch.Tensor) -> dict[str, float]:
        return {
            "euler_best_median": euler_score(generated, sampler, sim),
            "euler_best_median_real_val": real_euler,
            "hf_fraction_median": float(high_frequency_fraction(generated).median()),
            "hf_fraction_median_real_val": real_hf,
        }

    return metrics
