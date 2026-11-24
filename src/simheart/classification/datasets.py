"""Training and evaluation sets for the one-vs-rest augmentation experiments.

All beats (real and synthetic) are per-beat scaled into the simulator range and mapped to
[-1, 1], so the classifier sees every source on the same scale.

Setups for class c (proposal §7, plus oversampling from the team's implementation plan):

* ``real``:       real training beats only (the control)
* ``oversample``: real + real c beats repeated (drawn with replacement)
* ``simulator``, ``dcgan``, ``simdcgan``: real + synthetic c beats from that source

``amount`` is the number of added beats as a multiple of the real c training beats.
"""

from pathlib import Path

import numpy as np

from simheart.data.dataset import load_split
from simheart.data.labels import CLASS_TO_INDEX
from simheart.data.scaling import minmax_to_sim_range, sim_to_unit
from simheart.data.synthetic import ALL_METHODS, load_synthetic

SETUPS: tuple[str, ...] = ("real", "oversample", *ALL_METHODS)
SPLITS: tuple[str, ...] = ("train", "validation", "test")
STRICT_EXCLUDED_RECORDS: tuple[str, ...] = ("202",)  # same patient as training record 201


def real_binary(
    processed_dir: str | Path, split: str, beat_type: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """All real beats of a split: (beats [N, 216] on [-1, 1], 1 if class c else 0, records)."""
    data = load_split(processed_dir, split)
    beats = sim_to_unit(minmax_to_sim_range(data["X"]))
    labels = (data["y"] == CLASS_TO_INDEX[beat_type]).astype(np.int64)
    return beats, labels, data["record"]


def whole_database(
    processed_dir: str | Path, beat_type: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Every real beat (train + validation + test), as the SimGANs paper's Table 1 evaluates.

    Returns (beats, 0/1 labels, records, split): ``split`` names each beat's split
    ("train" = patients the classifier was trained on; "validation" and "test" = patients
    it never trained on).
    """
    parts = [real_binary(processed_dir, split, beat_type) for split in SPLITS]
    beats, labels, records = (np.concatenate(arrays) for arrays in zip(*parts, strict=True))
    split = np.concatenate(
        [np.full(len(p[1]), name) for name, p in zip(SPLITS, parts, strict=True)]
    )
    return beats, labels, records, split


def strict_mask(records: np.ndarray) -> np.ndarray:
    """True for beats kept in the strictly patient-separated test set."""
    return ~np.isin(records, STRICT_EXCLUDED_RECORDS)


def added_beats(
    processed_dir: str | Path,
    synthetic_dir: str | Path,
    beat_type: str,
    setup: str,
    amount: float,
    seed: int,
) -> np.ndarray:
    """The extra class-c beats a setup adds to the training set ([M, 216] on [-1, 1])."""
    if setup not in SETUPS:
        raise ValueError(f"setup must be one of {SETUPS}, got {setup!r}")
    if setup == "real":
        return np.zeros((0, 216), dtype=np.float32)
    beats, labels, _ = real_binary(processed_dir, "train", beat_type)
    positives = beats[labels == 1]
    if setup == "oversample":
        n = round(amount * len(positives))
        index = np.random.default_rng(seed).integers(0, len(positives), n)
        return positives[index]
    synthetic, _, _ = load_synthetic(synthetic_dir, setup, beat_type, seed, amount)
    return sim_to_unit(synthetic).astype(np.float32)


def augmented_training_set(
    processed_dir: str | Path,
    synthetic_dir: str | Path,
    beat_type: str,
    setup: str,
    amount: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Real training beats plus the setup's added class-c beats: (beats, 0/1 labels)."""
    beats, labels, _ = real_binary(processed_dir, "train", beat_type)
    extra = added_beats(processed_dir, synthetic_dir, beat_type, setup, amount, seed)
    return (
        np.concatenate([beats, extra]).astype(np.float32),
        np.concatenate([labels, np.ones(len(extra), dtype=np.int64)]),
    )


def synthetic_positive_set(
    processed_dir: str | Path,
    synthetic_dir: str | Path,
    beat_type: str,
    method: str,
    amount: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Train-on-synthetic set: real *negative* beats plus synthetic class-c beats only."""
    beats, labels, _ = real_binary(processed_dir, "train", beat_type)
    extra = added_beats(processed_dir, synthetic_dir, beat_type, method, amount, seed)
    negatives = beats[labels == 0]
    return (
        np.concatenate([negatives, extra]).astype(np.float32),
        np.concatenate([np.zeros(len(negatives), dtype=np.int64), np.ones(len(extra), np.int64)]),
    )
