import torch

from simheart.simulator import MCSHARRY_PARAMS, TYPICAL_PARAMS, EcgSyn
from simheart.simulator.fitting import (
    ParamKDE,
    ParamSpace,
    fit_beats,
    load_samplers,
    save_samplers,
)

SPACE = ParamSpace()


def assert_valid(params: torch.Tensor, space: ParamSpace = SPACE) -> None:
    assert (params[:, 1] >= space.b_min).all()
    assert (params[:, 2].diff(dim=1) > 0).all()  # θ_P < θ_Q < θ_R < θ_S < θ_T
    assert (params[:, 2, 2].abs() <= space.theta_r_max).all()


def test_any_raw_values_give_valid_parameters():
    raw = 3 * torch.randn(1000, 3, 5, generator=torch.Generator().manual_seed(0))
    assert_valid(SPACE.to_params(raw))


def test_raw_mapping_round_trips():
    params = torch.stack([*TYPICAL_PARAMS.values(), MCSHARRY_PARAMS])
    torch.testing.assert_close(SPACE.to_params(SPACE.to_raw(params)), params)


def test_fitting_recovers_beats_the_simulator_made():
    sim = EcgSyn()
    generator = torch.Generator().manual_seed(1)
    truth = torch.stack([TYPICAL_PARAMS["N"], TYPICAL_PARAMS["V"]])
    truth = truth * (1 + 0.1 * torch.randn(truth.shape, generator=generator))
    with torch.no_grad():
        beats = sim(truth)

    result = fit_beats(sim, beats, MCSHARRY_PARAMS[None], SPACE, steps=600, lr=0.03)

    assert (result.r2 > 0.99).all(), result.r2
    assert_valid(result.params)


def test_masked_samples_do_not_affect_the_fit():
    sim = EcgSyn()
    with torch.no_grad():
        beats = sim(TYPICAL_PARAMS["N"][None])
    corrupted = beats.clone()
    corrupted[:, -18:] += 1.0  # a large error only in the masked tail
    mask = torch.ones(216, dtype=torch.bool)
    mask[-18:] = False

    clean = fit_beats(sim, beats, TYPICAL_PARAMS["N"][None], SPACE, mask, steps=20)
    noisy = fit_beats(sim, corrupted, TYPICAL_PARAMS["N"][None], SPACE, mask, steps=20)

    torch.testing.assert_close(clean.params, noisy.params)
    torch.testing.assert_close(clean.mse, noisy.mse)


def fitted_set(n: int = 500, seed: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    """Raw parameters and z0 spread around the typical S beat, standing in for real fits."""
    generator = torch.Generator().manual_seed(seed)
    raw = SPACE.to_raw(TYPICAL_PARAMS["S"][None]) + 0.3 * torch.randn(n, 3, 5, generator=generator)
    z0 = 0.002 + 0.001 * torch.randn(n, generator=generator)
    return raw, z0


def test_sampler_without_jitter_returns_fitted_parameter_sets():
    raw, z0 = fitted_set()
    sampler = ParamKDE.fit(raw, z0, SPACE, jitter=0.0)

    params, z0_samples = sampler.sample(200, torch.Generator().manual_seed(3))

    fitted = SPACE.to_params(raw).reshape(len(raw), 15)
    # Exact pairwise distances (the default matrix-product shortcut loses float32 precision).
    distance = torch.cdist(
        params.reshape(200, 15), fitted, compute_mode="donot_use_mm_for_euclid_dist"
    )
    distance = distance.min(dim=1).values
    assert distance.max() < 1e-5
    assert set(z0_samples.tolist()) <= set(z0.tolist())


def test_sampler_jitter_scales_with_each_parameters_spread():
    raw, z0 = fitted_set()
    sampler = ParamKDE.fit(raw, z0, SPACE, jitter=0.02)
    spread = torch.cat([raw.reshape(len(raw), 15), z0[:, None]], dim=1).std(dim=0)
    torch.testing.assert_close(sampler.bandwidth, 0.02 * spread)

    params, z0_samples = sampler.sample(5000, torch.Generator().manual_seed(4))
    assert params.shape == (5000, 3, 5) and z0_samples.shape == (5000,)
    assert_valid(params)


def test_samplers_save_and_load(tmp_path):
    raw, z0 = fitted_set()
    sampler = ParamKDE.fit(raw, z0, SPACE)

    save_samplers(tmp_path / "samplers.npz", {"S": sampler})
    loaded = load_samplers(tmp_path / "samplers.npz")["S"]

    torch.testing.assert_close(loaded.values, sampler.values)
    torch.testing.assert_close(loaded.bandwidth, sampler.bandwidth)
    assert loaded.space == SPACE
