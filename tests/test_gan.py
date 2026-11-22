import numpy as np
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
