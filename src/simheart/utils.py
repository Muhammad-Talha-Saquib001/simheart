"""Shared helpers: configuration loading, seeding and device selection."""

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Read a YAML configuration file into a dictionary."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch so runs can be repeated.

    With ``deterministic=True`` cuDNN picks deterministic kernels, which is slightly
    slower but makes GPU results repeatable.
    """
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - seeds the global generator other libraries use
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(preferred: str | None = None) -> torch.device:
    """Return the requested device, or the first GPU if available, otherwise the CPU."""
    if preferred is not None:
        return torch.device(preferred)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
