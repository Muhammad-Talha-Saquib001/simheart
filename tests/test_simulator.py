import numpy as np
import pytest
import torch

from simheart.simulator import MCSHARRY_PARAMS, TYPICAL_PARAMS, EcgSyn, from_flat, to_flat


@pytest.fixture(scope="module")
def sim() -> EcgSyn:
    return EcgSyn()


def batch_of_params(n: int, seed: int = 0, dtype=torch.float32) -> torch.Tensor:
    """Typical-beat parameters with small random changes."""
    generator = torch.Generator().manual_seed(seed)
    base = torch.stack(list(TYPICAL_PARAMS.values()))
    base = base.repeat(n // len(TYPICAL_PARAMS) + 1, 1, 1)[:n]
    noise = 1 + 0.1 * torch.randn(base.shape, generator=generator)
    return (base * noise).to(dtype)


def test_output_shape_and_values_are_finite(sim):
    beats = sim(batch_of_params(8))
    assert beats.shape == (8, 216)
    assert torch.isfinite(beats).all()


def test_r_wave_is_the_peak_at_sample_72(sim):
    for params in [MCSHARRY_PARAMS, *TYPICAL_PARAMS.values()]:
        beat = sim(params[None])[0]
        assert abs(int(beat.argmax()) - 72) <= 2


def test_flat_conversion_round_trips_in_paper_order():
    params = batch_of_params(3)
    flat = to_flat(params)
    assert flat.shape == (3, 15)
    torch.testing.assert_close(flat[:, :5], params[:, 2])  # θ_P..θ_T come first
    torch.testing.assert_close(from_flat(flat), params)


def test_gradients_reach_the_parameters():
    sim = EcgSyn(n_samples=40).double()
    params = batch_of_params(2, dtype=torch.float64).requires_grad_()
    assert torch.autograd.gradcheck(sim, (params,), eps=1e-6, atol=1e-5)


def test_gradients_of_dz_dt_reach_the_signal_and_the_parameters():
    sim = EcgSyn(n_samples=40).double()
    params = batch_of_params(2, dtype=torch.float64).requires_grad_()
    z = (0.01 * torch.randn(2, 40, dtype=torch.float64)).requires_grad_()
    assert torch.autograd.gradcheck(sim.dz_dt, (z, params), eps=1e-6, atol=1e-5)


def test_simulated_beat_satisfies_its_own_equation():
    """A beat the simulator produced has zero Euler residual: the Euler loss of a perfect fit."""
    sim = EcgSyn().double()
    params = MCSHARRY_PARAMS[None].double()
    beat = sim(params)
    residual = (beat[:, 1:] - beat[:, :-1]) / sim.dt - sim.dz_dt(beat[:, :-1], params)
    assert residual.abs().max() < 1e-9


def test_matches_neurokit2_ecgsyn(sim):
    nk = pytest.importorskip("neurokit2")
    # 60 bpm at 216 Hz: one beat is exactly 216 samples, the same time base as EcgSyn.
    ecg = nk.ecg_simulate(
        duration=10,
        sampling_rate=216,
        heart_rate=60,
        heart_rate_std=0,
        noise=0,
        method="ecgsyn",
        random_state=0,
    )
    middle = 5 * 216
    peak = middle - 108 + int(np.argmax(ecg[middle - 108 : middle + 108]))
    reference = ecg[peak - 72 : peak + 144]
    ours = sim(MCSHARRY_PARAMS[None])[0].numpy()

    def minmax(v):
        return (v - v.min()) / (v.max() - v.min())

    assert np.corrcoef(minmax(reference), minmax(ours))[0, 1] > 0.98
