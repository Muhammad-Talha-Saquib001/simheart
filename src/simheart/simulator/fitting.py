"""Fitting ECGSYN parameters to real beats, and the per-class parameter distribution.

For each real beat we look for the 15 simulator parameters (plus the starting value z0)
whose simulated beat is closest to it. The Euler loss then needs parameter sets drawn
from each class's distribution of fitted parameters.

The optimiser works on unconstrained "raw" numbers that ``ParamSpace`` maps to valid
parameters, so every step gives a meaningful beat:

* widths:   b = b_min + softplus(raw)              -> always positive
* R angle:  θ_R = θ_R_max · tanh(raw)               -> the R wave stays near the R-peak
* others:   θ_Q = θ_R − softplus(raw), θ_P = θ_Q − softplus(raw),
            θ_S = θ_R + softplus(raw), θ_T = θ_S + softplus(raw)
                                                    -> P < Q < R < S < T, waves never swap
* amplitudes a are used as they are (either sign).

That distribution is modelled with a kernel density estimate (``ParamKDE``): a sample is
a randomly chosen fitted parameter set plus a small Gaussian jitter, in raw space so every
sample is valid. This is a deviation from the paper, which uses a diagonal Gaussian over
the parameters. The parameters relate to each other non-linearly (a wave's height depends
on amplitude × width²), so independent Gaussian draws mostly produce unrealistic beats:
on the training fits only 6-17 % of diagonal-Gaussian samples stayed within the real
amplitude range, against 89-91 % with this sampler at 2 % jitter.
"""

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from simheart.simulator.ecgsyn import A_ROW, B_ROW, THETA_ROW, EcgSyn

P, Q, R, S, T = range(5)


def _inverse_softplus(y: torch.Tensor) -> torch.Tensor:
    """Inverse of softplus for y > 0, written to stay accurate for small and large y."""
    return y + torch.log(-torch.expm1(-y))


@dataclass(frozen=True)
class ParamSpace:
    """Mapping between unconstrained raw tensors and valid [B, 3, 5] parameter tensors."""

    b_min: float = 0.02
    theta_r_max: float = math.radians(30)

    def to_params(self, raw: torch.Tensor) -> torch.Tensor:
        a = raw[:, A_ROW]
        b = self.b_min + F.softplus(raw[:, B_ROW])
        g = raw[:, THETA_ROW]
        theta_r = self.theta_r_max * torch.tanh(g[:, R])
        theta_q = theta_r - F.softplus(g[:, Q])
        theta_p = theta_q - F.softplus(g[:, P])
        theta_s = theta_r + F.softplus(g[:, S])
        theta_t = theta_s + F.softplus(g[:, T])
        theta = torch.stack([theta_p, theta_q, theta_r, theta_s, theta_t], dim=1)
        return torch.stack([a, b, theta], dim=1)

    def to_raw(self, params: torch.Tensor) -> torch.Tensor:
        """Inverse of ``to_params``. Needs b > b_min, ordered angles and |θ_R| < θ_R_max."""
        theta = params[:, THETA_ROW]
        gaps = torch.stack(
            [
                theta[:, Q] - theta[:, P],
                theta[:, R] - theta[:, Q],
                torch.atanh(theta[:, R] / self.theta_r_max),
                theta[:, S] - theta[:, R],
                theta[:, T] - theta[:, S],
            ],
            dim=1,
        )
        raw_theta = torch.cat(
            [_inverse_softplus(gaps[:, :2]), gaps[:, 2:3], _inverse_softplus(gaps[:, 3:])], dim=1
        )
        raw_b = _inverse_softplus(params[:, B_ROW] - self.b_min)
        return torch.stack([params[:, A_ROW], raw_b, raw_theta], dim=1)


@dataclass
class FitResult:
    """Best fit per beat over all starting points."""

    params: torch.Tensor  # [N, 3, 5]
    raw: torch.Tensor  # [N, 3, 5]
    z0: torch.Tensor  # [N]
    mse: torch.Tensor  # [N], over the masked samples
    r2: torch.Tensor  # [N], 1 - SSE/SST over the masked samples
    start_index: torch.Tensor  # [N], which starting point gave the best fit


def fit_beats(
    sim: EcgSyn,
    beats: torch.Tensor,
    starts: torch.Tensor,
    space: ParamSpace,
    mask: torch.Tensor | None = None,
    steps: int = 1500,
    lr: float = 0.02,
) -> FitResult:
    """Fit parameters to each beat from every starting point at once, keeping the best.

    ``beats`` is [N, L] on the simulator's device; ``starts`` is [K, 3, 5]. All K·N fits run
    as one batch. The loss is the squared error summed per beat (only where ``mask`` is
    True), so the fits are independent of each other.
    """
    n_beats, length = beats.shape
    n_starts = starts.shape[0]
    mask = torch.ones(length, dtype=torch.bool, device=beats.device) if mask is None else mask
    weight = mask.to(beats.dtype)

    target = beats.repeat(n_starts, 1)  # [K*N, L], start k occupies rows k*N .. (k+1)*N
    raw = space.to_raw(starts.to(beats)).repeat_interleave(n_beats, dim=0)
    raw = raw.clone().requires_grad_()
    z0 = target[:, 0].clone().requires_grad_()

    optimizer = torch.optim.Adam([raw, z0], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    for _ in range(steps):
        optimizer.zero_grad()
        error = ((sim(space.to_params(raw), z0) - target) ** 2 * weight).sum(dim=1)
        error.sum().backward()
        optimizer.step()
        scheduler.step()

    with torch.no_grad():
        params = space.to_params(raw)
        sse = ((sim(params, z0) - target) ** 2 * weight).sum(dim=1).view(n_starts, n_beats)
        best = sse.argmin(dim=0)
        rows = best * n_beats + torch.arange(n_beats, device=beats.device)
        n_used = weight.sum()
        mean = (beats * weight).sum(dim=1, keepdim=True) / n_used
        sst = ((beats - mean) ** 2 * weight).sum(dim=1)
        best_sse = sse.gather(0, best[None])[0]
        return FitResult(
            params=params[rows],
            raw=raw[rows].detach(),
            z0=z0[rows].detach(),
            mse=best_sse / n_used,
            r2=1 - best_sse / sst,
            start_index=best,
        )


@dataclass
class ParamKDE:
    """Kernel density estimate over one class's fitted raw parameters and z0.

    A sample is a randomly chosen fitted parameter set plus Gaussian jitter whose spread is
    ``jitter`` times each parameter's spread across the class.
    """

    values: torch.Tensor  # [M, 16]: 15 raw parameters and z0 per fitted beat
    bandwidth: torch.Tensor  # [16]
    space: ParamSpace

    @classmethod
    def fit(
        cls, raw: torch.Tensor, z0: torch.Tensor, space: ParamSpace, jitter: float = 0.02
    ) -> "ParamKDE":
        values = torch.cat([raw.reshape(len(raw), 15), z0[:, None]], dim=1)
        return cls(values, jitter * values.std(dim=0), space)

    def to(self, device: torch.device | str) -> "ParamKDE":
        return ParamKDE(self.values.to(device), self.bandwidth.to(device), self.space)

    def sample(
        self, n: int, generator: torch.Generator | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw n parameter sets: ([n, 3, 5] valid parameters, [n] z0)."""
        device = self.values.device
        index = torch.randint(len(self.values), (n,), generator=generator, device=device)
        noise = torch.randn(n, 16, generator=generator, device=device)
        drawn = self.values[index] + self.bandwidth * noise
        return self.space.to_params(drawn[:, :15].reshape(n, 3, 5)), drawn[:, 15]


@dataclass
class ParamGaussian:
    """The SimGANs paper's sampler: an independent Gaussian per raw parameter (and z0).

    Built from the same fitted values as a :class:`ParamKDE` (mean and sd of each raw
    parameter over the class). Used only for the Phase 15 ablation; it creates unrealistic
    parameter combinations (decision log §5), which is what the ablation tests.
    """

    mean: torch.Tensor  # [16]
    std: torch.Tensor  # [16]
    space: ParamSpace

    @classmethod
    def from_kde(cls, kde: ParamKDE) -> "ParamGaussian":
        return cls(kde.values.mean(dim=0), kde.values.std(dim=0), kde.space)

    def to(self, device: torch.device | str) -> "ParamGaussian":
        return ParamGaussian(self.mean.to(device), self.std.to(device), self.space)

    def sample(
        self, n: int, generator: torch.Generator | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw n parameter sets: ([n, 3, 5] valid parameters, [n] z0)."""
        noise = torch.randn(n, 16, generator=generator, device=self.mean.device)
        drawn = self.mean + self.std * noise
        return self.space.to_params(drawn[:, :15].reshape(n, 3, 5)), drawn[:, 15]


def save_samplers(path: str | Path, samplers: dict[str, ParamKDE]) -> None:
    """Save one sampler per class to a single .npz file."""
    spaces = {s.space for s in samplers.values()}
    if len(spaces) != 1:
        raise ValueError("All samplers must share one ParamSpace")
    space = spaces.pop()
    arrays: dict[str, np.ndarray] = {
        "classes": np.array(list(samplers)),
        "b_min": np.array(space.b_min),
        "theta_r_max": np.array(space.theta_r_max),
    }
    for name, s in samplers.items():
        arrays[f"{name}_values"] = s.values.cpu().numpy()
        arrays[f"{name}_bandwidth"] = s.bandwidth.cpu().numpy()
    np.savez(path, **arrays)


def load_samplers(path: str | Path) -> dict[str, ParamKDE]:
    """Load the samplers written by ``save_samplers``."""
    with np.load(path) as data:
        space = ParamSpace(float(data["b_min"]), float(data["theta_r_max"]))
        return {
            str(name): ParamKDE(
                torch.from_numpy(data[f"{name}_values"]),
                torch.from_numpy(data[f"{name}_bandwidth"]),
                space,
            )
            for name in data["classes"]
        }
