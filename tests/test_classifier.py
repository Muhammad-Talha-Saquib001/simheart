import json

import numpy as np
import pytest
import torch

from simheart.classification.datasets import (
    added_beats,
    augmented_training_set,
    real_binary,
    strict_mask,
    synthetic_positive_set,
)
from simheart.classification.metrics import (
    average_precision,
    best_f1_threshold,
    precision_at_recall,
    scores_at_threshold,
)
from simheart.classification.model import KachueeResNet
from simheart.classification.train import ClassifierConfig, predict_scores, train_classifier
from simheart.data.labels import CLASS_TO_INDEX
from simheart.data.scaling import minmax_to_sim_range
from simheart.data.synthetic import save_synthetic

CPU = torch.device("cpu")


@pytest.fixture(scope="module")
def dirs(tmp_path_factory):
    """A tiny prepared dataset (train 101/106, validation 118, test 100/202) and synthetic sets."""
    root = tmp_path_factory.mktemp("data")
    processed, synthetic = root / "processed", root / "synthetic"
    processed.mkdir()
    rng = np.random.default_rng(0)
    records = np.array(["101"] * 20 + ["106"] * 20 + ["118"] * 10 + ["100"] * 10 + ["202"] * 10)
    y = rng.integers(0, 4, len(records))
    y[:4] = CLASS_TO_INDEX["V"]  # make sure the training split has V beats
    np.savez(
        processed / "beats.npz",
        X=rng.normal(size=(len(records), 216)).astype(np.float32),
        y=y,
        record=records,
        patient=records,
        r_peak=np.arange(len(records)),
    )
    splits = {"train": ["101", "106"], "validation": ["118"], "test": ["100", "202"]}
    (processed / "splits.json").write_text(json.dumps({"records": splits}), encoding="utf-8")
    n_v = int(np.sum(y[:40] == CLASS_TO_INDEX["V"]))
    for method in ("simulator", "dcgan", "simdcgan"):
        beats = minmax_to_sim_range(rng.normal(size=(2 * n_v, 216)).astype(np.float32))
        save_synthetic(synthetic, method, "V", 0, beats, n_train=n_v, source="test")
    return processed, synthetic, n_v


def test_model_maps_beats_to_one_logit_each():
    model = KachueeResNet()
    assert model(torch.randn(3, 216)).shape == (3,)
    assert model(torch.randn(3, 1, 216)).shape == (3,)
    assert sum(p.numel() for p in model.parameters()) < 100_000


def test_training_learns_a_separable_toy_problem_and_fine_tuning_starts_from_the_state():
    rng = np.random.default_rng(0)
    time = np.linspace(-1, 1, 216)
    spike = np.exp(-((time / 0.05) ** 2))
    x = rng.normal(scale=0.1, size=(200, 216)).astype(np.float32)
    y = rng.integers(0, 2, 200)
    x[y == 1] += spike
    config = ClassifierConfig(lr=3e-3, batch_size=32)

    state, losses = train_classifier(x, y, 8, config, CPU, seed=0)
    assert losses[-1] < losses[0]
    assert average_precision(y, predict_scores(state, x, CPU)) > 0.95

    _, finetune_losses = train_classifier(x, y, 1, config, CPU, init_state=state, seed=1)
    assert finetune_losses[0] < losses[0]  # starts from the trained weights, not from scratch


def test_real_binary_labels_one_class_against_the_rest(dirs):
    processed, _, n_v = dirs
    x, y, records = real_binary(processed, "train", "V")
    assert x.shape == (40, 216) and y.sum() == n_v
    np.testing.assert_allclose(x.min(axis=1), -1, atol=1e-5)
    np.testing.assert_allclose(x.max(axis=1), 1, atol=1e-5)
    assert set(records) == {"101", "106"}


@pytest.mark.parametrize("setup", ["oversample", "simulator", "dcgan", "simdcgan"])
@pytest.mark.parametrize("amount", [0.5, 1.0, 2.0])
def test_augmented_sets_add_the_right_number_of_positive_beats(dirs, setup, amount):
    processed, synthetic, n_v = dirs
    x, y = augmented_training_set(processed, synthetic, "V", setup, amount, seed=0)
    assert len(x) == 40 + round(amount * n_v)
    assert y.sum() == n_v + round(amount * n_v)
    assert y[40:].all()


def test_control_adds_nothing_and_oversampling_repeats_only_real_positives(dirs):
    processed, synthetic, n_v = dirs
    x, y = augmented_training_set(processed, synthetic, "V", "real", 1.0, seed=0)
    assert len(x) == 40 and y.sum() == n_v
    real, labels, _ = real_binary(processed, "train", "V")
    repeated = added_beats(processed, synthetic, "V", "oversample", 2.0, seed=0)
    positives = {row.tobytes() for row in real[labels == 1]}
    assert all(row.tobytes() in positives for row in repeated)


def test_train_on_synthetic_set_has_no_real_positives(dirs):
    processed, synthetic, n_v = dirs
    x, y = synthetic_positive_set(processed, synthetic, "V", "simdcgan", 2.0, seed=0)
    assert y.sum() == 2 * n_v and (y == 0).sum() == 40 - n_v


def test_strict_mask_drops_record_202(dirs):
    processed, _, _ = dirs
    _, _, records = real_binary(processed, "test", "V")
    mask = strict_mask(records)
    assert not np.any(records[mask] == "202") and mask.sum() == 10


def test_metrics_match_hand_computed_values():
    labels = np.array([1, 0, 1, 0, 0, 1])
    scores = np.array([0.9, 0.8, 0.7, 0.4, 0.2, 0.1])
    # ranked: 1 0 1 0 0 1 -> precision at each positive: 1, 2/3, 3/6
    assert average_precision(labels, scores) == pytest.approx((1 + 2 / 3 + 0.5) / 3)
    assert precision_at_recall(labels, scores, 2 / 3) == pytest.approx(2 / 3)
    assert scores_at_threshold(labels, scores, 0.7) == pytest.approx(
        {"precision": 2 / 3, "recall": 2 / 3, "f1": 2 / 3}
    )
    threshold = best_f1_threshold(labels, scores)
    assert scores_at_threshold(labels, scores, threshold)["f1"] >= 2 / 3


def test_unknown_setup_is_rejected(dirs):
    processed, synthetic, _ = dirs
    with pytest.raises(ValueError, match="setup"):
        added_beats(processed, synthetic, "V", "vae", 1.0, seed=0)
