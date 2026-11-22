"""Checks against the real prepared data. Skipped until scripts/prepare_data.py has run."""

from pathlib import Path

import numpy as np
import pytest

from simheart.data.dataset import BEATS_FILE
from simheart.data.labels import CLASS_TO_INDEX
from simheart.data.splits import DS1

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"

# DS1 class totals reported by de Chazal et al. (2004), Table I.
DE_CHAZAL_DS1 = {"N": 45866, "S": 944, "V": 3788, "F": 415}

pytestmark = pytest.mark.skipif(
    not (PROCESSED / BEATS_FILE).exists(), reason="run scripts/prepare_data.py first"
)


def load():
    with np.load(PROCESSED / BEATS_FILE) as data:
        return {key: data[key] for key in data.files}


def test_ds1_class_totals_match_de_chazal_within_one_percent():
    data = load()
    ds1 = np.isin(data["record"], DS1)
    for name, expected in DE_CHAZAL_DS1.items():
        actual = int(np.sum(data["y"][ds1] == CLASS_TO_INDEX[name]))
        assert abs(actual - expected) <= 0.01 * expected, f"{name}: {actual} vs {expected}"


def test_beats_are_finite_and_216_samples_long():
    data = load()
    assert data["X"].shape[1] == 216
    assert np.isfinite(data["X"]).all()
