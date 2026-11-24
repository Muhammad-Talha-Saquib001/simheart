import numpy as np
import pytest
import torch

from simheart.models import Discriminator, Generator, add_spectral_norm, init_weights
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


def test_spectral_norm_discriminator_drops_batch_norm_and_normalises_weights():
    torch.manual_seed(0)
    plain = Discriminator()
    normed = Discriminator(spectral_norm=True)
    assert any(isinstance(m, torch.nn.BatchNorm1d) for m in plain.net)
    assert not any(isinstance(m, torch.nn.BatchNorm1d) for m in normed.net)
    normed.apply(init_weights)
    add_spectral_norm(normed)
    for _ in range(20):  # power iteration converges during training-mode forward passes
        normed(torch.randn(4, 1, 216))
    assert normed(torch.randn(3, 1, 216)).shape == (3,)
    for module in normed.net:
        if isinstance(module, torch.nn.Conv1d):
            weight = module.weight.reshape(module.weight.shape[0], -1)
            assert torch.linalg.matrix_norm(weight, ord=2).item() == pytest.approx(1.0, abs=0.05)


def test_stabilised_training_runs_and_label_smoothing_changes_only_the_discriminator(tmp_path):
    time = np.linspace(-1, 1, 216)
    beats = np.exp(-((time / 0.1) ** 2))[None] * np.random.default_rng(0).uniform(0.5, 1, (64, 1))
    logs = {}
    for real_label in (1.0, 0.9):
        torch.manual_seed(0)
        config = GanTrainConfig(
            iterations=1, batch_size=8, log_every=1, eval_every=1, snapshot_every=1, n_eval=16,
            discriminator_spectral_norm=True, real_label=real_label,
        )  # fmt: skip
        result = train_gan(
            beats[:48], beats[48:], config, torch.device("cpu"), tmp_path / str(real_label)
        )
        logs[real_label] = result.train_log[0]
        assert load_generator(tmp_path / str(real_label) / "best.pt") is not None
    # Same start and same batches: the discriminator loss differs, D's outputs don't.
    assert logs[0.9]["loss_d"] != logs[1.0]["loss_d"]
    assert logs[0.9]["d_real"] == pytest.approx(logs[1.0]["d_real"])
    assert GanTrainConfig().real_label == 1.0 and not GanTrainConfig().discriminator_spectral_norm
