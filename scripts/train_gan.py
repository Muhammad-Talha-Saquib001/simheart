"""Train a GAN for one beat class and save checkpoints, logs and figures.

Run from the repository root, after scripts/prepare_data.py:

    .venv/bin/python scripts/train_gan.py --config configs/dcgan.yaml --beat-type V

Reads:  the config, data/processed/{beats.npz, splits.json}
Writes: checkpoints/<model>/<class>/{best,last}.pt
        outputs/runs/<model>_<class>/{train_log,eval_log}.csv, curves.png, progress.png,
        final_samples.png, summary.json
"""

import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from simheart.data.dataset import load_split
from simheart.data.mitbih import SAMPLING_RATE
from simheart.data.scaling import minmax_to_sim_range, sim_to_unit
from simheart.evaluation.physiology import physiology_metrics
from simheart.losses.euler import PAIRINGS, EulerLoss
from simheart.models.gan import NOISE_DIM
from simheart.simulator import EcgSyn
from simheart.simulator.fitting import load_samplers
from simheart.training.gan import GanTrainConfig, generate, load_generator, train_gan
from simheart.utils import get_device, load_config, set_seed

ROOT = Path(__file__).resolve().parents[1]
R_PEAK_INDEX = 72
TIME_MS = (np.arange(216) - R_PEAK_INDEX) / SAMPLING_RATE * 1000


def load_class(processed_dir: Path, split: str, beat_type: str) -> np.ndarray:
    """Beats of one class and split, scaled to the networks' [-1, 1] range."""
    beats = load_split(processed_dir, split, classes=[beat_type])["X"]
    return sim_to_unit(minmax_to_sim_range(beats))


def write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def smooth(values: np.ndarray, window: int = 20) -> np.ndarray:
    return np.convolve(values, np.ones(window) / window, mode="valid")


def plot_curves(result, title: str, path: Path) -> None:
    train = {k: np.array([row[k] for row in result.train_log]) for k in result.train_log[0]}
    evals = {k: np.array([row[k] for row in result.eval_log]) for k in result.eval_log[0]}
    fig, axes = plt.subplots(1, 4, figsize=(21, 4))
    it = train["iteration"][19:]

    axes[0].plot(it, smooth(train["loss_d"]), label="discriminator")
    axes[0].plot(it, smooth(train["loss_g_adversarial"]), label="generator (adversarial)")
    if np.any(train["loss_g_extra"]):
        axes[0].plot(it, smooth(train["loss_g_extra"]), label="generator (extra)")
    axes[0].set_title("Losses (moving average of 20 logs)")
    axes[0].legend(fontsize=8)

    axes[1].plot(it, smooth(train["d_real"]), label="D(real)")
    axes[1].plot(it, smooth(train["d_fake"]), label="D(generated)")
    axes[1].axhline(0.5, color="grey", linestyle=":")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Discriminator output (probability of 'real')")
    axes[1].legend(fontsize=8)

    ax = axes[2]
    ax.plot(evals["iteration"], evals["mmd"], "o-", color="tab:red", label="MMD to validation")
    ax.axhline(
        evals["mmd_real_train_reference"][0],
        color="k",
        linestyle="--",
        label="real training beats (reference)",
    )
    ax.axvline(result.best_iteration, color="tab:red", linestyle=":", label="best checkpoint")
    ax.set_ylabel("MMD (lower is more realistic)")
    ax.set_yscale("log")
    twin = ax.twinx()
    twin.plot(
        evals["iteration"], evals["diversity_ratio"], "s-", color="tab:blue", label="diversity"
    )
    twin.axhline(1, color="tab:blue", linestyle=":")
    twin.set_ylabel("diversity ratio (1 = as varied as real)")
    twin.set_ylim(0, 1.5)
    ax.set_title("Validation metrics")
    lines = ax.get_legend_handles_labels()
    twin_lines = twin.get_legend_handles_labels()
    ax.legend(lines[0] + twin_lines[0], lines[1] + twin_lines[1], fontsize=8)

    ax = axes[3]
    ax.plot(evals["iteration"], evals["euler_best_median"], "o-", color="tab:green", label="Euler")
    ax.axhline(evals["euler_best_median_real_val"][0], color="tab:green", linestyle="--",
               label="Euler, real validation")  # fmt: skip
    ax.set_ylabel("best-of-32 Euler distance (median)")
    twin = ax.twinx()
    twin.plot(evals["iteration"], evals["hf_fraction_median"], "s-", color="tab:purple",
              label="power > 40 Hz")  # fmt: skip
    twin.axhline(evals["hf_fraction_median_real_val"][0], color="tab:purple", linestyle=":",
                 label="power > 40 Hz, real")  # fmt: skip
    twin.set_ylabel("share of power above 40 Hz")
    ax.set_title("Physiology and smoothness")
    lines = ax.get_legend_handles_labels()
    twin_lines = twin.get_legend_handles_labels()
    ax.legend(lines[0] + twin_lines[0], lines[1] + twin_lines[1], fontsize=8)
    for a in axes:
        a.set_xlabel("iteration")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_progress(snapshots: dict[int, np.ndarray], title: str, path: Path) -> None:
    """The same four noise vectors at up to five points during training."""
    steps = sorted(snapshots)
    steps = [steps[i] for i in np.linspace(0, len(steps) - 1, min(5, len(steps))).astype(int)]
    fig, axes = plt.subplots(len(steps), 4, figsize=(13, 2.1 * len(steps)), sharex=True)
    axes = np.atleast_2d(axes)
    for row, step in enumerate(steps):
        for col in range(4):
            axes[row, col].plot(TIME_MS, snapshots[step][col], linewidth=1)
        axes[row, 0].set_ylabel(f"iter {step}")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def plot_final(generated: np.ndarray, real_val: np.ndarray, title: str, path: Path) -> None:
    low, mid, high = np.percentile(real_val, [10, 50, 90], axis=0)
    g_low, g_mid, g_high = np.percentile(generated, [10, 50, 90], axis=0)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharey=True)
    for ax, beats, name in [
        (axes[0], generated, "generated"),
        (axes[1], real_val, "real (validation)"),
    ]:
        ax.fill_between(TIME_MS, low, high, color="tab:blue", alpha=0.15, label="real 10-90%")
        ax.plot(
            TIME_MS,
            beats[:20].T,
            color="tab:red" if name == "generated" else "k",
            lw=0.7,
            alpha=0.6,
        )
        ax.set_title(f"20 {name} beats")
        ax.legend(fontsize=8)
    axes[2].fill_between(TIME_MS, low, high, color="tab:blue", alpha=0.2, label="real 10-90%")
    axes[2].plot(TIME_MS, mid, color="tab:blue", label="real median")
    axes[2].fill_between(
        TIME_MS, g_low, g_high, color="tab:red", alpha=0.2, label="generated 10-90%"
    )
    axes[2].plot(TIME_MS, g_mid, color="tab:red", label="generated median")
    axes[2].set_title("Distribution per time step")
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.set_xlabel("time from R-peak (ms)")
    axes[0].set_ylabel("amplitude ([-1, 1] scale)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/dcgan.yaml", type=Path)
    parser.add_argument("--beat-type", choices=["S", "V", "F"])
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--weight", type=float, help="Euler-loss weight λ (0 = plain DCGAN)")
    parser.add_argument("--pairing", choices=PAIRINGS)
    parser.add_argument("--run-name", help="name for the checkpoint and output folders")
    args = parser.parse_args()

    config = load_config(ROOT / args.config)
    beat_type = args.beat_type or config["beat_type"]
    seed = config["seed"] if args.seed is None else args.seed
    training = dict(config["training"], betas=tuple(config["training"]["betas"]))
    if args.iterations:
        training["iterations"] = args.iterations
    train_config = GanTrainConfig(**training)
    euler = dict(config.get("euler", {"weight": 0.0}))
    if args.weight is not None:
        euler["weight"] = args.weight
    if args.pairing:
        euler["pairing"] = args.pairing

    model = config["model"]
    run_name = args.run_name or model
    processed_dir = ROOT / config["processed_dir"]
    checkpoint_dir = ROOT / config["checkpoints_dir"] / run_name / beat_type
    run_dir = ROOT / config["runs_dir"] / f"{run_name}_{beat_type}"
    run_dir.mkdir(parents=True, exist_ok=True)

    set_seed(seed)
    device = get_device()
    beats_train = load_class(processed_dir, "train", beat_type)
    beats_val = load_class(processed_dir, "validation", beat_type)
    print(f"{run_name} {beat_type}: {len(beats_train)} training, {len(beats_val)} validation beats")

    sim = EcgSyn().to(device)
    sampler = load_samplers(ROOT / config["samplers_path"])[beat_type].to(device)
    metrics = physiology_metrics(sim, sampler, torch.as_tensor(beats_val, device=device))
    euler_loss = None
    if euler["weight"] > 0:
        euler_loss = EulerLoss(
            sim, sampler, euler["weight"], euler.get("pairing", "random"), euler.get("k", 16)
        )
        print(f"Euler loss: weight {euler['weight']}, pairing {euler_loss.pairing}")

    began = time.time()
    result = train_gan(
        beats_train,
        beats_val,
        train_config,
        device,
        checkpoint_dir,
        extra_generator_loss=euler_loss,
        extra_metrics=metrics,
        run_info={
            "model": model, "run_name": run_name, "beat_type": beat_type, "seed": seed,
            "euler": euler,
        },
    )  # fmt: skip
    minutes = (time.time() - began) / 60

    write_csv(run_dir / "train_log.csv", result.train_log)
    write_csv(run_dir / "eval_log.csv", result.eval_log)
    title = f"{run_name} {beat_type} ({len(beats_train)} training beats)"
    plot_curves(result, title, run_dir / "curves.png")
    plot_progress(
        result.snapshots, f"{title}: fixed noise during training", run_dir / "progress.png"
    )

    generator = load_generator(checkpoint_dir / "best.pt", device)
    generated = generate(generator, torch.randn(512, NOISE_DIM, device=device)).cpu().numpy()
    plot_final(
        generated, beats_val, f"{title}: best checkpoint (iteration {result.best_iteration})",
        run_dir / "final_samples.png",
    )  # fmt: skip

    best = next(row for row in result.eval_log if row["iteration"] == result.best_iteration)
    summary = {
        "model": model,
        "run_name": run_name,
        "beat_type": beat_type,
        "seed": seed,
        "euler": euler,
        "training_beats": len(beats_train),
        "validation_beats": len(beats_val),
        "iterations": train_config.iterations,
        "minutes": round(minutes, 1),
        "best": best,
        "last": result.eval_log[-1],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
