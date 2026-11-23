"""Small helpers for the results pipeline: pre-registered choices and report tables.

The two choice rules are fixed in advance and use validation data only, so the test set
(DS2) never influences a decision:

* ``choose_pilot_settings``: the classifier's learning rate and number of base epochs.
* ``default_seeds``: which trained generator (seed) the user interface uses by default.
"""

from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

# SimGANs paper, Table 1: precision at fixed recall, evaluated on the whole MIT-BIH database.
PAPER_TABLE_1 = {
    "S": {"recall": 0.41, "dcgan": 0.58, "simdcgan": 0.80},
    "V": {"recall": 0.91, "dcgan": 0.72, "simdcgan": 0.84},
    "F": {"recall": 0.60, "dcgan": 0.20, "simdcgan": 0.40},
}

# Phase 12 failure rule, fixed before the reliability runs (decision log §19).
FAILURE_MIN_DIVERSITY = 0.3
FAILURE_MAX_HF_FRACTION = 0.10

FINETUNE_EPOCHS = 5


def mean_std(values: Iterable[float]) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=float)
    return float(array.mean()), float(array.std())


def choose_pilot_settings(rows: Sequence[dict]) -> dict:
    """Pick the classifier's learning rate and base epochs from validation-only pilot rows.

    Rows: {"class", "lr", "epoch", "val_ap"}. Rule: the learning rate whose best validation
    AP, averaged over classes, is highest; then, at that rate, the epoch with the highest
    validation AP averaged over classes. Fine-tuning always uses ``FINETUNE_EPOCHS``.
    """
    by_lr_class: dict[tuple[float, str], list[dict]] = defaultdict(list)
    for row in rows:
        by_lr_class[(float(row["lr"]), row["class"])].append(row)
    lrs = sorted({lr for lr, _ in by_lr_class})
    classes = sorted({c for _, c in by_lr_class})

    def best_ap(lr: float) -> float:
        return float(
            np.mean([max(float(r["val_ap"]) for r in by_lr_class[(lr, c)]) for c in classes])
        )

    lr = max(lrs, key=best_ap)
    per_epoch: dict[int, list[float]] = defaultdict(list)
    for c in classes:
        for row in by_lr_class[(lr, c)]:
            per_epoch[int(row["epoch"])].append(float(row["val_ap"]))
    complete = {e: v for e, v in per_epoch.items() if len(v) == len(classes)}
    epochs = max(complete, key=lambda e: (np.mean(complete[e]), -e))  # ties: fewer epochs
    return {
        "lr": lr,
        "base_epochs": epochs,
        "finetune_epochs": FINETUNE_EPOCHS,
        "rule": (
            "learning rate with the highest mean (over classes) best validation AP; then the "
            "epoch with the highest mean validation AP at that rate"
        ),
        "evidence": {
            "best_val_ap_by_lr": {str(k): best_ap(k) for k in lrs},
            "mean_val_ap_at_chosen_epoch": float(np.mean(complete[epochs])),
            "classes": classes,
        },
    }


def default_seeds(eval_rows: Sequence[dict], models: Sequence[str] = ("dcgan", "simdcgan")) -> dict:
    """For each (model, class), the seed whose generator has the lowest final MMD to validation.

    Rows: {"class", "method", "seed", "mmd"} (the per-seed generator evaluation).
    """
    best: dict[tuple[str, str], tuple[float, int]] = {}
    for row in eval_rows:
        if row["method"] not in models:
            continue
        key, mmd, seed = (row["method"], row["class"]), float(row["mmd"]), int(row["seed"])
        if key not in best or mmd < best[key][0]:
            best[key] = (mmd, seed)
    return {key: seed for key, (_, seed) in best.items()}


def missing_runs(
    runs_dir: str | Path,
    classes: Sequence[str] = ("S", "V", "F"),
    models: Sequence[str] = ("dcgan", "simdcgan"),
    seeds: Sequence[int] = (0, 1, 2),
) -> list[str]:
    """Training runs (``<model>_s<seed>_<class>``) without a summary.json, i.e. not finished."""
    runs = Path(runs_dir)
    names = [f"{m}_s{s}_{c}" for c in classes for s in seeds for m in models]
    return [name for name in names if not (runs / name / "summary.json").exists()]


def markdown_table(
    rows: Sequence[dict],
    columns: Sequence[tuple[str, str]],
    formats: dict[str, str] | None = None,
) -> str:
    """A Markdown table. ``columns`` are (key, header) pairs; ``formats`` maps key -> format."""
    formats = formats or {}

    def text(row: dict, key: str) -> str:
        value = row.get(key, "")
        if key in formats and isinstance(value, int | float | np.floating):
            return format(value, formats[key])
        return str(value)

    lines = [
        "| " + " | ".join(header for _, header in columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    lines += ["| " + " | ".join(text(row, key) for key, _ in columns) + " |" for row in rows]
    return "\n".join(lines)


def is_failed_run(best: dict) -> bool:
    """Phase 12 rule: a GAN run failed if its best checkpoint is collapsed or noise-like.

    ``best`` is the ``best`` entry of a run's summary.json. Failed means a diversity ratio
    below 0.3 (near-identical beats) or more than 10 % of signal power above 40 Hz (noise).
    """
    return (
        best["diversity_ratio"] < FAILURE_MIN_DIVERSITY
        or best["hf_fraction_median"] > FAILURE_MAX_HF_FRACTION
    )


def clopper_pearson(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Exact (Clopper-Pearson) interval for a rate of k events in n trials."""
    from scipy.stats import beta

    alpha = 1 - level
    low = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    high = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return low, high
