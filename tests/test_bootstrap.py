import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from simheart.classification.bootstrap import (
    bootstrap_metrics,
    interval,
    record_weights,
    weighted_metrics,
)
from simheart.classification.metrics import precision_at_recall


def make_test_set(seed=0, n_records=10, per_record=60):
    rng = np.random.default_rng(seed)
    records = np.repeat([f"r{i}" for i in range(n_records)], per_record)
    labels = (rng.random(len(records)) < 0.2).astype(int)
    labels[:3] = 1  # at least some positives
    good = labels + rng.normal(0, 0.6, len(labels))
    bad = labels + rng.normal(0, 1.5, len(labels))
    return records, labels, good, bad


def test_replicates_keep_whole_records_and_draw_n_records():
    records, *_ = make_test_set()
    counts, beat_record = record_weights(records, 100, np.random.default_rng(0))
    assert counts.shape == (100, 10)
    assert (counts.sum(axis=1) == 10).all()  # 10 records drawn per replicate
    weights = counts[:, beat_record]
    for r in range(10):  # every beat of a record has the same weight
        block = weights[:, beat_record == r]
        assert (block == block[:, :1]).all()


def test_weighted_metrics_match_sklearn_on_duplicated_beats():
    records, labels, good, _ = make_test_set()
    good = np.round(good, 1)  # create ties on purpose
    counts, beat_record = record_weights(records, 5, np.random.default_rng(1))
    weights = counts[:, beat_record]
    ap, p_at_r = weighted_metrics(labels, good, weights, recall=0.6)
    for b in range(5):
        idx = np.repeat(np.arange(len(labels)), weights[b])
        assert ap[b] == pytest.approx(average_precision_score(labels[idx], good[idx]))
        assert p_at_r[b] == pytest.approx(precision_at_recall(labels[idx], good[idx], 0.6))


def test_identical_setups_differ_by_exactly_zero_and_better_setup_wins():
    records, labels, good, bad = make_test_set()
    scores = {"a": np.stack([good, good]), "b": np.stack([good, good]), "bad": bad[None]}
    points, samples = bootstrap_metrics(labels, scores, records, 0.5, n_boot=300)
    assert np.array_equal(samples["a"]["ap"], samples["b"]["ap"])
    diff = interval(samples["a"]["ap"] - samples["bad"]["ap"], 0.0)
    assert diff["low"] > 0 and diff["above_0"] == 1.0
    assert points["a"]["ap"] == pytest.approx(average_precision_score(labels, good))


def test_replicates_without_positives_are_skipped_and_counted():
    records, labels, good, _ = make_test_set()
    labels = np.zeros_like(labels)
    labels[records == "r0"] = 1  # all positives in one record
    _, samples = bootstrap_metrics(labels, {"a": good[None]}, records, 0.5, n_boot=500)
    result = interval(samples["a"]["ap"], 0.0)
    missing = np.isnan(samples["a"]["ap"]).sum()
    assert missing > 0  # record r0 is left out in about (9/10)^10 = 35 % of replicates
    assert result["usable"] == 500 - missing


def test_same_seed_gives_same_replicates():
    records, labels, good, _ = make_test_set()
    one = bootstrap_metrics(labels, {"a": good[None]}, records, 0.5, n_boot=50, seed=3)[1]
    two = bootstrap_metrics(labels, {"a": good[None]}, records, 0.5, n_boot=50, seed=3)[1]
    assert np.array_equal(one["a"]["ap"], two["a"]["ap"])
