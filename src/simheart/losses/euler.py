"""The Euler loss of SimGANs (Golany et al., ICML 2020).

For a beat h (in simulator units) and simulator parameters η, the simulator distance is

    Δ_sim(h, η) = Σ_{ℓ=1}^{L−1} ( (h_{ℓ+1} − h_ℓ)/Δt − f_z(θ_ℓ, h_ℓ, t_ℓ; η) )²

the squared mismatch between the beat's own sample-to-sample change and the change the
ECGSYN equation predicts at that point. It is 0 when the beat is exactly what the
simulator would produce with η. The generator's Euler loss is the average of Δ_sim over
generated beats, multiplied by a weight λ.

How η is paired with each generated beat:

* ``random``    (the paper's formula): one η drawn from the class's parameter sampler.
                Even real beats score about 20 this way, and minimising it pulls all
                beats towards one "average" physiology.
* ``best_of_k`` (the paper's stated intuition, "lie close to the submanifold of signals
                the simulator can produce"): k η are drawn and the smallest Δ_sim is used,
                so a beat is penalised only if no plausible heart setting explains it.
"""

import torch
from torch import nn

from simheart.data.scaling import unit_to_sim
from simheart.simulator.ecgsyn import EcgSyn
from simheart.simulator.fitting import ParamKDE

PAIRINGS = ("random", "best_of_k")


def euler_distance(beats_sim: torch.Tensor, params: torch.Tensor, sim: EcgSyn) -> torch.Tensor:
    """Δ_sim for each beat [B, L] (simulator units) with its parameters [B, 3, 5]: [B]."""
    change = (beats_sim[:, 1:] - beats_sim[:, :-1]) / sim.dt
    residual = change - sim.dz_dt(beats_sim[:, :-1], params)
    return (residual**2).sum(dim=1)


def best_euler_distance(
    beats_sim: torch.Tensor,
    sampler: ParamKDE,
    sim: EcgSyn,
    k: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Smallest Δ_sim of each beat over k parameter sets drawn from ``sampler``: [B]."""
    n = len(beats_sim)
    params, _ = sampler.sample(n * k, generator)
    distances = euler_distance(beats_sim.repeat_interleave(k, dim=0), params, sim)
    return distances.view(n, k).min(dim=1).values


class EulerLoss(nn.Module):
    """λ · mean Δ_sim of generated beats, which arrive on the networks' [-1, 1] scale."""

    def __init__(
        self,
        sim: EcgSyn,
        sampler: ParamKDE,
        weight: float,
        pairing: str = "random",
        k: int = 16,
    ) -> None:
        super().__init__()
        if pairing not in PAIRINGS:
            raise ValueError(f"pairing must be one of {PAIRINGS}, got {pairing!r}")
        self.sim, self.sampler = sim, sampler
        self.weight, self.pairing, self.k = weight, pairing, k

    def forward(self, generated_unit: torch.Tensor) -> torch.Tensor:
        beats = unit_to_sim(generated_unit.reshape(len(generated_unit), -1))
        if self.pairing == "random":
            params, _ = self.sampler.sample(len(beats))
            distances = euler_distance(beats, params, self.sim)
        else:
            distances = best_euler_distance(beats, self.sampler, self.sim, self.k)
        return self.weight * distances.mean()
