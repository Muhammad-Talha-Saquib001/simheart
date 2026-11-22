"""Sample-quality measures that need no classifier: realism, diversity and copying.

All functions take beats as tensors shaped [N, L] (or [N, 1, L]) on any device.
"""

import torch


def _flat(beats: torch.Tensor) -> torch.Tensor:
    return beats.reshape(len(beats), -1).double()


def _distances(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # Exact pairwise distances (the default matrix-product shortcut loses precision).
    return torch.cdist(a, b, compute_mode="donot_use_mm_for_euclid_dist")


def mmd_rbf(x: torch.Tensor, y: torch.Tensor, bandwidth: float | None = None) -> float:
    """Squared maximum mean discrepancy between two sets, with a Gaussian (RBF) kernel.

    Zero when the two sets come from the same distribution; larger means more different.
    By default the kernel width is the median distance between all pooled beats (the
    "median heuristic"). This is the unbiased estimate, so it can be slightly negative.
    """
    x, y = _flat(x), _flat(y)
    xx, yy, xy = _distances(x, x), _distances(y, y), _distances(x, y)
    if bandwidth is None:
        pooled = torch.cat([xx.flatten(), yy.flatten(), xy.flatten()])
        bandwidth = float(pooled[pooled > 0].median())

    def kernel(d: torch.Tensor) -> torch.Tensor:
        return torch.exp(-(d**2) / (2 * bandwidth**2))

    n, m = len(x), len(y)
    k_xx = (kernel(xx).sum() - n) / (n * (n - 1))  # leave out the diagonal (distance 0)
    k_yy = (kernel(yy).sum() - m) / (m * (m - 1))
    return float(k_xx + k_yy - 2 * kernel(xy).mean())


def mean_pairwise_distance(beats: torch.Tensor) -> float:
    """Average distance between two different beats of a set: 0 means all beats are identical."""
    flat = _flat(beats)
    n = len(flat)
    return float(_distances(flat, flat).sum() / (n * (n - 1)))


def nearest_distances(queries: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Distance from each query beat to its closest beat in the reference set: [N_queries]."""
    return _distances(_flat(queries), _flat(reference)).min(dim=1).values


def high_frequency_fraction(
    beats: torch.Tensor, sampling_rate: float = 360.0, cutoff_hz: float = 40.0
) -> torch.Tensor:
    """Share of each beat's signal power above ``cutoff_hz`` (mean removed): [N].

    ECG waves sit mostly below 40 Hz; a higher share means a more jagged, noisier beat.
    """
    flat = _flat(beats)
    power = torch.fft.rfft(flat - flat.mean(dim=1, keepdim=True), dim=1).abs() ** 2
    freqs = torch.fft.rfftfreq(flat.shape[1], d=1 / sampling_rate).to(flat.device)
    return power[:, freqs > cutoff_hz].sum(dim=1) / power[:, 1:].sum(dim=1).clamp_min(1e-12)


def near_copy_rate(
    generated: torch.Tensor,
    train: torch.Tensor,
    real_val: torch.Tensor,
    quantile: float = 0.05,
) -> float:
    """Share of generated beats that are suspiciously close to a training beat.

    The threshold is the ``quantile`` of the distances from real validation beats (other
    patients, never trained on) to their nearest training beat. Real held-out beats
    therefore score ``quantile`` (5 %) by construction; a generator that memorises training
    beats scores higher.
    """
    threshold = torch.quantile(nearest_distances(real_val, train), quantile)
    return float((nearest_distances(generated, train) < threshold).double().mean())
