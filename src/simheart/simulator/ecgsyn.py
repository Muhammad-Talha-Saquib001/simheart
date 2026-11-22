"""The ECGSYN heart model (McSharry et al., 2003) as a batched, differentiable PyTorch module.

The model moves a point (x, y) around the unit circle; one lap is one heartbeat. As the
point passes five fixed angles θ_P … θ_T, the ECG value z is pushed up or down, which
draws the P, Q, R, S and T waves:

    dx/dt = α x − ω y,    dy/dt = α y + ω x,    α = 1 − sqrt(x² + y²)
    dz/dt = −Σ_i a_i Δθ_i exp(−Δθ_i² / 2b_i²) − (z − z_base(t))
    Δθ_i = (atan2(y, x) − θ_i) wrapped to (−π, π],    z_base(t) = A sin(2π f₂ t)

Each wave has an amplitude a, a width b and an angle θ: 15 parameters per beat, held here
as a tensor shaped [batch, 3, 5] (rows a, b, θ; columns P, Q, R, S, T).

Choices that differ from the SimGANs paper's text (see docs/references-and-resources.md):

* Time base: step h = 1/216 and ω = 2π, so the 216-sample window is exactly one cycle
  (as in the authors' code). The paper's h = 1/360 would cover only 60 % of a cycle.
* The start angle puts the R wave (θ = 0) at sample ``r_peak_index`` (72), where the
  R-peak sits in the real beats.
* Δθ is wrapped to (−π, π] (as in NeuroKit2) rather than ``fmod``, which jumps where the
  angle wraps at ±180° and would break gradients there.
* Baseline wander A = 0.005, as in the authors' code and NeuroKit2 (the paper says 0.15 mV).

The (x, y) path does not depend on the parameters or on z, so it is integrated once with
Euler steps and cached; only z is stepped per batch.
"""

import math

import torch
from torch import nn

WAVES: tuple[str, ...] = ("P", "Q", "R", "S", "T")
A_ROW, B_ROW, THETA_ROW = 0, 1, 2


def make_params(
    a: tuple[float, ...], b: tuple[float, ...], theta_degrees: tuple[float, ...]
) -> torch.Tensor:
    """Build a [3, 5] parameter tensor from per-wave amplitudes, widths and angles (degrees)."""
    theta = [math.radians(d) for d in theta_degrees]
    return torch.tensor([a, b, theta], dtype=torch.float32)


# Normal beat from McSharry et al. (2003), Table 1.
MCSHARRY_PARAMS = make_params(
    a=(1.2, -5.0, 30.0, -7.5, 0.75),
    b=(0.25, 0.1, 0.1, 0.1, 0.4),
    theta_degrees=(-70.0, -15.0, 0.0, 15.0, 100.0),
)

# Starting values per class from the SimGANs authors' code
# (DreamStudioAI/sim_gan, sim_gan/dynamical_model/typical_beat_params.py), reordered.
TYPICAL_PARAMS: dict[str, torch.Tensor] = {
    "N": make_params(
        a=(0.7, -7.0, 30.0, -3.0, 0.2),
        b=(0.25, 0.1, 0.1, 0.1, 0.4),
        theta_degrees=(-90.0, -15.0, 0.0, 15.0, 160.0),
    ),
    "S": make_params(
        a=(0.2, -1.0, 30.0, -10.0, 0.2),
        b=(0.25, 0.1, 0.1, 0.1, 0.4),
        theta_degrees=(-90.0, -15.0, 0.0, 15.0, 160.0),
    ),
    "V": make_params(
        a=(0.1, 0.0, 30.0, -10.0, 0.5),
        b=(0.6, 0.1, 0.1, 0.1, 0.2),
        theta_degrees=(-90.0, -15.0, 0.0, 15.0, 160.0),
    ),
    "F": make_params(
        a=(0.8, -10.0, 30.0, -10.0, 0.5),
        b=(0.25, 0.1, 0.1, 0.1, 0.2),
        theta_degrees=(-90.0, -15.0, 0.03, 15.0, 160.0),
    ),
}


def to_flat(params: torch.Tensor) -> torch.Tensor:
    """[B, 3, 5] -> [B, 15] in the paper's order (θ_P..θ_T, a_P..a_T, b_P..b_T)."""
    return torch.cat([params[:, THETA_ROW], params[:, A_ROW], params[:, B_ROW]], dim=1)


def from_flat(flat: torch.Tensor) -> torch.Tensor:
    """[B, 15] in the paper's order -> [B, 3, 5]."""
    theta, a, b = flat.reshape(-1, 3, 5).unbind(dim=1)
    return torch.stack([a, b, theta], dim=1)


class EcgSyn(nn.Module):
    """ECGSYN simulator producing beats of ``n_samples`` samples, one cycle per beat."""

    def __init__(
        self,
        n_samples: int = 216,
        dt: float = 1 / 216,
        omega: float = 2 * math.pi,
        r_peak_index: int = 72,
        baseline_amplitude: float = 0.005,
        respiration_hz: float = 0.25,
    ) -> None:
        super().__init__()
        self.n_samples = n_samples
        self.dt = dt
        theta = self._phase_path(n_samples, dt, omega, start=-omega * dt * r_peak_index)
        t = torch.arange(n_samples, dtype=torch.float64) * dt
        z_base = baseline_amplitude * torch.sin(2 * math.pi * respiration_hz * t)
        self.register_buffer("theta", theta.float())
        self.register_buffer("z_base", z_base.float())

    @staticmethod
    def _phase_path(n_samples: int, dt: float, omega: float, start: float) -> torch.Tensor:
        """Integrate the (x, y) equations with Euler steps and return the angle at each sample."""
        x, y = math.cos(start), math.sin(start)
        angles = []
        for _ in range(n_samples):
            angles.append(math.atan2(y, x))
            alpha = 1.0 - math.hypot(x, y)
            x, y = x + dt * (alpha * x - omega * y), y + dt * (alpha * y + omega * x)
        return torch.tensor(angles, dtype=torch.float64)

    def wave_drive(self, params: torch.Tensor, length: int | None = None) -> torch.Tensor:
        """The wave term −Σ_i a_i Δθ_i exp(−Δθ_i² / 2b_i²) at each sample: [B, length]."""
        length = self.n_samples if length is None else length
        a, b, theta_i = (params[:, row, :, None] for row in (A_ROW, B_ROW, THETA_ROW))
        diff = self.theta[:length] - theta_i  # [B, 5, length]
        delta = torch.atan2(torch.sin(diff), torch.cos(diff))
        return -(a * delta * torch.exp(-(delta**2) / (2 * b**2))).sum(dim=1)

    def dz_dt(self, z: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """Right-hand side of the z equation for a given signal ``z`` [B, L'] (L' <= n_samples).

        Sample ℓ of ``z`` is paired with the model's angle and time at step ℓ. This is the
        quantity the Euler loss compares with the signal's finite differences.
        """
        length = z.shape[-1]
        return self.wave_drive(params, length) - (z - self.z_base[:length])

    def forward(self, params: torch.Tensor, z0: float | torch.Tensor = 0.0) -> torch.Tensor:
        """Simulate one beat per parameter set with Euler steps: [B, 3, 5] -> [B, n_samples]."""
        drive = self.wave_drive(params)
        z = torch.as_tensor(z0, dtype=params.dtype, device=params.device).expand(params.shape[0])
        samples = [z]
        for step in range(self.n_samples - 1):
            z = z + self.dt * (drive[:, step] - (z - self.z_base[step]))
            samples.append(z)
        return torch.stack(samples, dim=1)
