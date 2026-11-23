"""Training loop shared by the plain DCGAN and SimDCGAN.

Both models use the same networks, optimiser and schedule. SimDCGAN only adds a term to
the generator's loss, passed in as ``extra_generator_loss``, so the two runs differ in
exactly one ingredient.

Losses (Goodfellow et al., 2014; as in the SimGANs paper):

* discriminator: −log D(x) − log(1 − D(G(z)))
* generator:     −log D(G(z))  (+ the extra term, if any)

computed from logits with ``BCEWithLogitsLoss`` for numerical stability.
"""

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

from simheart.evaluation.metrics import mean_pairwise_distance, mmd_rbf, nearest_distances
from simheart.models.gan import NOISE_DIM, Discriminator, Generator, init_weights

ExtraLoss = Callable[[torch.Tensor], torch.Tensor]
ExtraMetrics = Callable[[torch.Tensor], dict[str, float]]


@dataclass
class GanTrainConfig:
    iterations: int = 5000
    batch_size: int = 64
    lr: float = 2e-4
    betas: tuple[float, float] = (0.5, 0.999)
    g_steps_per_d_step: int = 2
    log_every: int = 10
    eval_every: int = 250
    snapshot_every: int = 500
    n_eval: int = 512
    n_snapshot: int = 16
    # Warm-up for the extra generator loss: its weight rises linearly from 0 to 1 over the
    # first ``extra_warmup`` iterations (0 = full weight from the start, the default).
    extra_warmup: int = 0


def warmup_factor(iteration: int, warmup: int) -> float:
    """Multiplier for the extra loss at ``iteration`` (1-based): iteration / warmup, capped at 1."""
    return 1.0 if warmup <= 0 else min(1.0, iteration / warmup)


@dataclass
class GanTrainResult:
    train_log: list[dict] = field(default_factory=list)
    eval_log: list[dict] = field(default_factory=list)
    snapshots: dict[int, np.ndarray] = field(default_factory=dict)  # iteration -> [16, 216]
    best_iteration: int = 0
    best_mmd: float = float("inf")


def load_generator(path: str | Path, device: torch.device | str = "cpu") -> Generator:
    """Load the generator from a checkpoint written by ``train_gan``, in evaluation mode."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    generator = Generator().to(device)
    generator.load_state_dict(checkpoint["generator"])
    return generator.eval()


@torch.no_grad()
def generate(generator: Generator, noise: torch.Tensor) -> torch.Tensor:
    """Run the generator in evaluation mode and return beats shaped [N, 216]."""
    was_training = generator.training
    generator.eval()
    beats = generator(noise).squeeze(1)
    generator.train(was_training)
    return beats


def train_gan(
    beats_train: np.ndarray,
    beats_val: np.ndarray,
    config: GanTrainConfig,
    device: torch.device,
    checkpoint_dir: Path,
    extra_generator_loss: ExtraLoss | None = None,
    extra_metrics: ExtraMetrics | None = None,
    run_info: dict | None = None,
) -> GanTrainResult:
    """Train a generator on ``beats_train`` ([N, 216], already on the [-1, 1] scale).

    Every ``eval_every`` iterations the generator is scored against ``beats_val``
    (validation beats, never trained on): MMD for realism, a diversity ratio, and
    nearest-training-beat distances for copying, plus whatever ``extra_metrics`` returns
    for the generated beats. The checkpoint with the lowest MMD is saved as ``best.pt``;
    the final one as ``last.pt``. ``run_info`` is stored in both.
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    real_train = torch.as_tensor(beats_train, dtype=torch.float32, device=device)
    real_val = torch.as_tensor(beats_val, dtype=torch.float32, device=device)

    generator, discriminator = Generator().to(device), Discriminator().to(device)
    generator.apply(init_weights)
    discriminator.apply(init_weights)
    opt_g = torch.optim.Adam(generator.parameters(), lr=config.lr, betas=config.betas)
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=config.lr, betas=config.betas)
    bce = nn.BCEWithLogitsLoss()

    snapshot_noise = torch.randn(config.n_snapshot, NOISE_DIM, device=device)
    eval_noise = torch.randn(config.n_eval, NOISE_DIM, device=device)
    real_diversity = mean_pairwise_distance(real_val)
    real_nn_median = float(nearest_distances(real_val, real_train).median())
    # Reference: how far real training beats are from the validation beats. Validation beats
    # come from other patients, so even a perfect copy of the training distribution scores
    # about this much.
    reference_index = torch.randperm(len(real_train), device=device)[: config.n_eval]
    real_mmd_reference = mmd_rbf(real_train[reference_index], real_val)

    result = GanTrainResult()
    stored = {"config": asdict(config), "run_info": run_info or {}}

    def save(name: str, iteration: int, metrics: dict) -> None:
        torch.save(
            {
                "generator": generator.state_dict(),
                "discriminator": discriminator.state_dict(),
                "iteration": iteration,
                "metrics": metrics,
                **stored,
            },
            checkpoint_dir / name,
        )

    def noise(n: int) -> torch.Tensor:
        return torch.randn(n, NOISE_DIM, device=device)

    for iteration in range(1, config.iterations + 1):
        real = real_train[torch.randint(len(real_train), (config.batch_size,), device=device)]
        ones = torch.ones(config.batch_size, device=device)
        zeros = torch.zeros(config.batch_size, device=device)

        # Discriminator: real beats -> 1, generated beats -> 0.
        opt_d.zero_grad()
        logits_real = discriminator(real)
        logits_fake = discriminator(generator(noise(config.batch_size)).detach())
        loss_d = bce(logits_real, ones) + bce(logits_fake, zeros)
        loss_d.backward()
        opt_d.step()

        # Generator: make the discriminator call generated beats real (fresh noise each step).
        for _ in range(config.g_steps_per_d_step):
            opt_g.zero_grad()
            fake = generator(noise(config.batch_size))
            logits_gen = discriminator(fake)
            loss_adv = bce(logits_gen, ones)
            extra = (
                extra_generator_loss(fake)
                if extra_generator_loss
                else torch.zeros((), device=device)
            )
            if config.extra_warmup > 0:
                extra = warmup_factor(iteration, config.extra_warmup) * extra
            (loss_adv + extra).backward()
            opt_g.step()

        if iteration % config.log_every == 0:
            result.train_log.append(
                {
                    "iteration": iteration,
                    "loss_d": loss_d.item(),
                    "loss_g_adversarial": loss_adv.item(),
                    "loss_g_extra": extra.item(),
                    "extra_factor": warmup_factor(iteration, config.extra_warmup),
                    "d_real": torch.sigmoid(logits_real).mean().item(),
                    "d_fake": torch.sigmoid(logits_fake).mean().item(),
                }
            )

        if iteration % config.snapshot_every == 0 or iteration == config.iterations:
            result.snapshots[iteration] = generate(generator, snapshot_noise).cpu().numpy()

        if iteration % config.eval_every == 0 or iteration == config.iterations:
            fake_eval = generate(generator, eval_noise)
            metrics = {
                "iteration": iteration,
                "mmd": mmd_rbf(fake_eval, real_val),
                "mmd_real_train_reference": real_mmd_reference,
                "diversity_ratio": mean_pairwise_distance(fake_eval) / real_diversity,
                "nn_train_median": float(nearest_distances(fake_eval, real_train).median()),
                "nn_train_median_real_val": real_nn_median,
            }
            if extra_metrics is not None:
                metrics.update(extra_metrics(fake_eval))
            result.eval_log.append(metrics)
            if metrics["mmd"] < result.best_mmd:
                result.best_mmd, result.best_iteration = metrics["mmd"], iteration
                save("best.pt", iteration, metrics)

    save("last.pt", config.iterations, result.eval_log[-1])
    return result
