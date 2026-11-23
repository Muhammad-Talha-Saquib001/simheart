import json

import numpy as np
import pytest
import torch

from simheart.data.labels import CLASS_TO_INDEX
from simheart.data.scaling import minmax_to_sim_range
from simheart.data.synthetic import load_synthetic, save_synthetic
from simheart.evaluation import near_copy_rate


def test_saved_set_returns_the_requested_amount_with_labels_and_metadata(tmp_path):
    beats = minmax_to_sim_range(np.random.default_rng(0).normal(size=(40, 216)).astype(np.float32))
    save_synthetic(tmp_path, "simdcgan", "F", 1, beats, n_train=20, source="test.pt")

    for amount, expected in [(0.5, 10), (1.0, 20), (2.0, 40)]:
        x, y, metadata = load_synthetic(tmp_path, "simdcgan", "F", 1, amount)
        assert x.shape == (expected, 216)
        np.testing.assert_array_equal(x, beats[:expected])  # smaller amounts are prefixes
        assert (y == CLASS_TO_INDEX["F"]).all()
    assert metadata["synthetic"] is True
    assert metadata["method"] == "simdcgan" and metadata["seed"] == 1
    json.dumps(metadata)


def test_asking_for_more_than_was_saved_fails(tmp_path):
    save_synthetic(tmp_path, "dcgan", "S", 0, np.zeros((10, 216)), n_train=5, source="x")
    with pytest.raises(ValueError, match="amount"):
        load_synthetic(tmp_path, "dcgan", "S", 0, amount=3.0)


def test_unknown_method_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="method"):
        save_synthetic(tmp_path, "vae", "S", 0, np.zeros((2, 216)), n_train=1, source="x")


def test_near_copy_rate_is_about_five_percent_for_new_draws_and_one_for_copies():
    generator = torch.Generator().manual_seed(0)
    train = torch.randn(500, 216, generator=generator)
    # New draws from the same distribution score about the quantile; copies score 1.
    fresh = torch.randn(400, 216, generator=generator)
    assert near_copy_rate(fresh, train) == pytest.approx(0.05, abs=0.05)
    assert near_copy_rate(train[:100], train) == 1.0
    noisy_copies = train[:100] + 0.01 * torch.randn(100, 216, generator=generator)
    assert near_copy_rate(noisy_copies, train) == 1.0
