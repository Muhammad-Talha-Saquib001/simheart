import numpy as np
import pytest
import torch

from simheart.models import Discriminator, Generator, init_weights
from simheart.training import GanTrainConfig, generate, load_generator, train_gan


def test_generator_maps_noise_to_one_beat():
    generator = Generator()
    assert generator(torch.randn(3, 100)).shape == (3, 1, 216)


def test_discriminator_returns_one_logit_per_beat():
    assert Discriminator()(torch.randn(3, 1, 216)).shape == (3,)


def test_dcgan_initialisation():
    generator = Generator()
    generator.apply(init_weights)
    first = generator.net[0].weight
    assert abs(first.mean().item()) < 1e-3
    assert abs(first.std().item() - 0.02) < 1e-3


def test_training_smoke_run_writes_loadable_checkpoints(tmp_path):
    torch.manual_seed(0)
    time = np.linspace(-1, 1, 216)
    beats = np.exp(-((time / 0.1) ** 2))[None] * np.random.default_rng(0).uniform(0.5, 1, (64, 1))
    config = GanTrainConfig(
        iterations=6, batch_size=8, log_every=2, eval_every=3, snapshot_every=3, n_eval=16
    )

    calls = []

    def extra(fake):
        calls.append(fake.shape)
        return fake.pow(2).mean()

    result = train_gan(
        beats[:48], beats[48:], config, torch.device("cpu"), tmp_path, extra_generator_loss=extra
    )

    assert len(calls) == config.iterations * config.g_steps_per_d_step
    assert all(np.isfinite(row["loss_d"]) for row in result.train_log)
    assert any(row["loss_g_extra"] > 0 for row in result.train_log)
    assert [row["iteration"] for row in result.eval_log] == [3, 6]
    assert sorted(result.snapshots) == [3, 6]
    assert (tmp_path / "best.pt").exists() and (tmp_path / "last.pt").exists()

    generator = load_generator(tmp_path / "last.pt")
    noise = torch.randn(4, 100)
    assert generate(generator, noise).shape == (4, 216)
    torch.testing.assert_close(generate(generator, noise), generate(generator, noise))


def test_warmup_ramps_the_extra_loss_from_zero_to_full_weight():
    from simheart.training.gan import warmup_factor

    assert [warmup_factor(i, 4) for i in (1, 2, 4, 5, 100)] == [0.25, 0.5, 1.0, 1.0, 1.0]
    assert warmup_factor(1, 0) == 1.0  # no warm-up: full weight from the start


def test_warmup_scales_the_logged_extra_loss(tmp_path):
    torch.manual_seed(0)
    beats = np.random.default_rng(0).normal(size=(40, 216))
    config = GanTrainConfig(
        iterations=8, batch_size=8, log_every=1, eval_every=8, snapshot_every=8, n_eval=8,
        extra_warmup=4,
    )  # fmt: skip
    result = train_gan(
        beats[:32], beats[32:], config, torch.device("cpu"), tmp_path,
        extra_generator_loss=lambda fake: torch.ones((), device=fake.device),
    )  # fmt: skip
    logged = [row["loss_g_extra"] for row in result.train_log]
    assert logged == pytest.approx([0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0, 1.0])
    assert [row["extra_factor"] for row in result.train_log][:4] == [0.25, 0.5, 0.75, 1.0]
