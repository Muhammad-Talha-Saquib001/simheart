"""Saving and loading synthetic beat sets for the augmentation experiments.

Each set holds synthetic beats of one class, made by one method (``simulator``, ``dcgan``
or ``simdcgan``) with one seed, at twice the class's real training size. Smaller amounts
are the first part of the set: amount 0.5 gives the first quarter, 1.0 the first half.

Beats are stored like real beats after loading: scaled per beat with
``minmax_to_sim_range``, so amplitude range cannot tell synthetic from real beats. Every
file is marked ``synthetic = True`` with its source (proposal §12: synthetic data must
never be presented as real patient data).
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from simheart.data.labels import CLASS_TO_INDEX

METHODS: tuple[str, ...] = ("simulator", "dcgan", "simdcgan")


def synthetic_path(synthetic_dir: str | Path, method: str, beat_type: str, seed: int) -> Path:
    return Path(synthetic_dir) / method / f"{beat_type}_s{seed}.npz"


def save_synthetic(
    synthetic_dir: str | Path,
    method: str,
    beat_type: str,
    seed: int,
    beats: np.ndarray,
    n_train: int,
    source: str,
) -> Path:
    """Save scaled synthetic beats [n, 216] with labels and metadata; returns the path."""
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    metadata = {
        "synthetic": True,
        "method": method,
        "beat_type": beat_type,
        "seed": seed,
        "n_train_real": n_train,
        "source": source,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "note": "SYNTHETIC heartbeats for research only; not real patient data.",
    }
    path = synthetic_path(synthetic_dir, method, beat_type, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        X=beats.astype(np.float32),
        y=np.full(len(beats), CLASS_TO_INDEX[beat_type], dtype=np.int64),
        metadata=np.array(json.dumps(metadata)),
    )
    return path


def load_synthetic(
    synthetic_dir: str | Path,
    method: str,
    beat_type: str,
    seed: int,
    amount: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return (beats, labels, metadata) for ``amount`` × the class's real training size."""
    with np.load(synthetic_path(synthetic_dir, method, beat_type, seed)) as data:
        metadata = json.loads(str(data["metadata"]))
        n = round(amount * metadata["n_train_real"])
        if n > len(data["X"]):
            raise ValueError(f"amount {amount} needs {n} beats; the set has {len(data['X'])}")
        return data["X"][:n], data["y"][:n], metadata
