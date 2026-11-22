import pytest
import torch

from simheart.data.scaling import sim_to_unit
from simheart.losses import EulerLoss, best_euler_distance, euler_distance
from simheart.simulator import TYPICAL_PARAMS, EcgSyn
from simheart.simulator.fitting import ParamKDE, ParamSpace

SPACE = ParamSpace()


@pytest.fixture(scope="module")
def sim() -> EcgSyn:
    return EcgSyn()


def class_sampler(name: str = "V", n: int = 200, spread: float = 0.2) -> ParamKDE:
    """A sampler of parameter sets spread around one class's typical beat."""
    generator = torch.Generator().manual_seed(0)
    raw = SPACE.to_raw(TYPICAL_PARAMS[name][None]) + spread * torch.randn(
        n, 3, 5, generator=generator
    )
    return ParamKDE.fit(raw, torch.zeros(n), SPACE)


def test_distance_is_zero_for_the_simulators_own_beat(sim):
    params = torch.stack([TYPICAL_PARAMS["S"], TYPICAL_PARAMS["V"]])
    with torch.no_grad():
        beats = sim(params)
    assert euler_distance(beats, params, sim).max() < 1e-6


def test_distance_is_positive_for_mismatched_parameters(sim):
    with torch.no_grad():
        beats = sim(TYPICAL_PARAMS["V"][None])
    assert euler_distance(beats, TYPICAL_PARAMS["N"][None], sim).item() > 1.0


def test_best_of_k_returns_the_smallest_distance_over_its_k_draws(sim):
    sampler = class_sampler()
    n, k = 8, 16
    with torch.no_grad():
        beats = sim(sampler.sample(n, torch.Generator().manual_seed(1))[0])

    best = best_euler_distance(beats, sampler, sim, k, torch.Generator().manual_seed(2))

    # Repeat the same draws by hand: beat i is compared with draws i*k .. i*k + k - 1.
    params, _ = sampler.sample(n * k, torch.Generator().manual_seed(2))
    every = euler_distance(beats.repeat_interleave(k, dim=0), params, sim).view(n, k)
    torch.testing.assert_close(best, every.min(dim=1).values)
    assert torch.all(best <= every[:, 0])  # never worse than a single draw


@pytest.mark.parametrize("pairing", ["random", "best_of_k"])
def test_loss_reads_unit_scale_beats_and_sends_gradients_to_them(sim, pairing):
    sampler = class_sampler()
    with torch.no_grad():
        beats_sim = sim(TYPICAL_PARAMS["V"][None].repeat(4, 1, 1))
    generated = sim_to_unit(beats_sim).unsqueeze(1).requires_grad_()

    loss = EulerLoss(sim, sampler, weight=0.5, pairing=pairing, k=4)(generated)
    loss.backward()

    assert loss.ndim == 0 and loss.item() > 0
    assert generated.grad is not None and generated.grad.abs().sum() > 0


def test_weight_scales_the_loss_linearly(sim):
    sampler = class_sampler()
    generated = sim_to_unit(sim(TYPICAL_PARAMS["V"][None]).detach()).repeat(4, 1)
    torch.manual_seed(3)
    small = EulerLoss(sim, sampler, weight=0.1)(generated)
    torch.manual_seed(3)
    large = EulerLoss(sim, sampler, weight=1.0)(generated)
    torch.testing.assert_close(large, 10 * small)


def test_unknown_pairing_is_rejected(sim):
    with pytest.raises(ValueError, match="pairing"):
        EulerLoss(sim, class_sampler(), weight=1.0, pairing="nearest")
