import numpy as np
import torch

from simheart.data.scaling import (
    SIM_MAX,
    SIM_MIN,
    minmax_to_sim_range,
    sim_to_unit,
    unit_to_sim,
)


def test_each_beat_is_scaled_into_the_simulator_range():
    rng = np.random.default_rng(0)
    beats = (rng.normal(size=(5, 216)) * rng.uniform(0.1, 3, size=(5, 1))).astype(np.float32)

    scaled = minmax_to_sim_range(beats)

    assert scaled.dtype == np.float32
    np.testing.assert_allclose(scaled.min(axis=1), SIM_MIN, atol=1e-7)
    np.testing.assert_allclose(scaled.max(axis=1), SIM_MAX, atol=1e-7)


def test_flat_beat_does_not_divide_by_zero():
    scaled = minmax_to_sim_range(np.zeros((1, 216), dtype=np.float32))
    assert np.isfinite(scaled).all()


def test_unit_scale_round_trips_and_spans_minus_one_to_one():
    beats = minmax_to_sim_range(np.random.default_rng(1).normal(size=(4, 216)).astype(np.float32))
    unit = sim_to_unit(beats)
    np.testing.assert_allclose(unit.min(axis=1), -1, atol=1e-5)
    np.testing.assert_allclose(unit.max(axis=1), 1, atol=1e-5)
    np.testing.assert_allclose(unit_to_sim(unit), beats, atol=1e-7)
    tensor = torch.as_tensor(unit)
    torch.testing.assert_close(unit_to_sim(tensor), torch.as_tensor(beats))
