"""Patient-level (cluster) bootstrap for detection metrics on the test set.

Beats of one patient are very alike, so the bootstrap resamples whole test *records* with
replacement, not single beats. A replicate is a weight per beat: how many times its record
was drawn. Weighted metrics equal the metrics on a test set where those beats are copied
that many times. Every setup is scored on the same replicates, so differences between
setups are paired.

The metrics are computed for all replicates at once: each score vector is sorted once
(the order doesn't depend on the weights), and the weighted true- and false-positive counts
come from cumulative sums along that order.
"""

from collections.abc import Mapping

import numpy as np


def record_weights(
    records: np.ndarray, n_boot: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Draw ``n_boot`` replicates of the test records, with replacement.

    Returns ``counts`` [n_boot, n_records] (how often each record was drawn) and
    ``beat_record`` [N] (each beat's record index), so a replicate's beat weights are
    ``counts[b, beat_record]``.
    """
    unique, beat_record = np.unique(records, return_inverse=True)
    n = len(unique)
    draws = rng.integers(0, n, size=(n_boot, n))
    counts = np.stack([np.bincount(row, minlength=n) for row in draws])
    return counts, beat_record


def weighted_metrics(
    labels: np.ndarray, scores: np.ndarray, weights: np.ndarray, recall: float
) -> tuple[np.ndarray, np.ndarray]:
    """Average precision and precision at ``recall`` for each row of ``weights`` [b, N].

    Same definitions as sklearn's ``average_precision_score`` and the highest precision
    reaching ``recall`` on ``precision_recall_curve``. Rows without any positive weight
    give NaN.
    """
    order = np.argsort(-scores, kind="stable")
    sorted_scores, positive = scores[order], labels[order].astype(bool)
    # Thresholds sit at the last beat of each group of equal scores.
    last = np.r_[np.flatnonzero(np.diff(sorted_scores) != 0), len(sorted_scores) - 1]
    w = weights[:, order].astype(np.float64)
    tp = np.cumsum(w * positive, axis=1)[:, last]
    fp = np.cumsum(w * ~positive, axis=1)[:, last]
    total = tp[:, -1:]
    with np.errstate(invalid="ignore", divide="ignore"):
        precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        rec = tp / total
    step = np.diff(np.concatenate([np.zeros_like(total), rec], axis=1), axis=1)
    ap = np.sum(step * precision, axis=1)
    p_at_r = np.max(np.where(rec >= recall - 1e-12, precision, -np.inf), axis=1)
    no_positive = total[:, 0] <= 0
    ap[no_positive] = np.nan
    p_at_r[no_positive] = np.nan
    return ap, p_at_r


def bootstrap_metrics(
    labels: np.ndarray,
    scores: Mapping[str, np.ndarray],
    records: np.ndarray,
    recall: float,
    n_boot: int = 2000,
    seed: int = 0,
    chunk: int = 200,
) -> tuple[dict, dict]:
    """Point values and bootstrap replicates for every setup, averaged over seeds.

    ``scores`` maps a setup name to an array [n_seeds, N]. Returns ``(points, samples)``:
    ``points[name] = {"ap": float, "p_at_r": float}`` on the real test set, and
    ``samples[name] = {"ap": [n_boot], "p_at_r": [n_boot]}`` (NaN where a replicate has
    no positive beats). All setups share the same replicates.
    """
    counts, beat_record = record_weights(records, n_boot, np.random.default_rng(seed))
    ones = np.ones((1, len(labels)))
    points, samples = {}, {}
    for name, per_seed in scores.items():
        point = [weighted_metrics(labels, s, ones, recall) for s in per_seed]
        points[name] = {
            "ap": float(np.mean([p[0][0] for p in point])),
            "p_at_r": float(np.mean([p[1][0] for p in point])),
        }
        ap, p_at_r = np.zeros((len(per_seed), n_boot)), np.zeros((len(per_seed), n_boot))
        for start in range(0, n_boot, chunk):
            weights = counts[start : start + chunk][:, beat_record]
            for i, s in enumerate(per_seed):
                a, p = weighted_metrics(labels, s, weights, recall)
                ap[i, start : start + chunk], p_at_r[i, start : start + chunk] = a, p
        samples[name] = {"ap": ap.mean(axis=0), "p_at_r": p_at_r.mean(axis=0)}
    return points, samples


def interval(samples: np.ndarray, point: float, level: float = 0.95) -> dict:
    """Percentile interval of the usable (non-NaN) replicates.

    Also returns the share of usable replicates above 0 (meaningful for differences) and
    how many replicates were usable.
    """
    usable = samples[~np.isnan(samples)]
    if len(usable) == 0:
        return {"point": point, "low": np.nan, "high": np.nan, "above_0": np.nan, "usable": 0}
    tail = (1 - level) / 2 * 100
    low, high = np.percentile(usable, [tail, 100 - tail])
    return {
        "point": point,
        "low": float(low),
        "high": float(high),
        "above_0": float(np.mean(usable > 0)),
        "usable": int(len(usable)),
    }
