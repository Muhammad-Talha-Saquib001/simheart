"""Loading the prepared beats by split, and wrapping them for PyTorch."""

import json
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from simheart.data.labels import CLASS_TO_INDEX

BEATS_FILE = "beats.npz"
SPLITS_FILE = "splits.json"


def load_split(
    processed_dir: str | Path,
    split: str,
    classes: Sequence[str] | None = None,
) -> dict[str, np.ndarray]:
    """Return the beats of one split ("train", "validation" or "test").

    The result has the arrays saved by ``scripts/prepare_data.py`` (``X`` in millivolts,
    ``y``, ``record``, ``patient``, ``r_peak``), filtered to the split's records and,
    optionally, to the given classes.
    """
    processed_dir = Path(processed_dir)
    splits = json.loads((processed_dir / SPLITS_FILE).read_text(encoding="utf-8"))
    if split not in splits["records"]:
        raise KeyError(f"Unknown split {split!r}; expected one of {list(splits['records'])}")

    with np.load(processed_dir / BEATS_FILE) as data:
        arrays = {key: data[key] for key in data.files}

    mask = np.isin(arrays["record"], splits["records"][split])
    if classes is not None:
        mask &= np.isin(arrays["y"], [CLASS_TO_INDEX[c] for c in classes])
    return {key: value[mask] for key, value in arrays.items()}


class BeatDataset(Dataset):
    """Beats as float tensors shaped [1, length], with their class index."""

    def __init__(
        self,
        beats: np.ndarray,
        labels: np.ndarray,
        transform: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        if transform is not None:
            beats = transform(beats)
        self.beats = torch.as_tensor(beats, dtype=torch.float32).unsqueeze(1)
        self.labels = torch.as_tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.beats[index], self.labels[index]
