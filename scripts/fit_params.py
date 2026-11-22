"""Fit ECGSYN parameters to each class's training beats and save per-class parameter samplers.

Run from the repository root, after scripts/prepare_data.py:

    .venv/bin/python scripts/fit_params.py

Reads:  configs/fitting.yaml, data/processed/{beats.npz, splits.json}
Writes: data/processed/{fitted_params.npz, param_samplers.npz}
        outputs/tables/{fit_quality, fitted_param_summary}.csv
        outputs/figures/fits_{S,V,F}.png
"""

import argparse
import csv
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from simheart.data.dataset import load_split
from simheart.data.mitbih import SAMPLING_RATE
from simheart.data.scaling import minmax_to_sim_range
from simheart.simulator import MCSHARRY_PARAMS, TYPICAL_PARAMS, WAVES, EcgSyn
from simheart.simulator.fitting import ParamKDE, ParamSpace, fit_beats, save_samplers
from simheart.utils import get_device, load_config, set_seed

ROOT = Path(__file__).resolve().parents[1]
START_NAMES = [*(f"typical {c}" for c in TYPICAL_PARAMS), "McSharry"]
R_PEAK_INDEX = 72
N_SAMPLES_SHOWN = 20


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def plot_class(
    name: str,
    beats: np.ndarray,
    fitted: np.ndarray,
    r2: np.ndarray,
    accepted: np.ndarray,
    min_r2: float,
    samples: np.ndarray,
    path: Path,
) -> None:
    """Best, median and worst accepted fits, the R² histogram, and beats from the sampler."""
    time_ms = (np.arange(beats.shape[1]) - R_PEAK_INDEX) / SAMPLING_RATE * 1000
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    order = np.flatnonzero(accepted)[np.argsort(r2[accepted])]
    picks = {"worst": order[0], "median": order[len(order) // 2], "best": order[-1]}
    for (label, i), color in zip(
        picks.items(), ["tab:red", "tab:orange", "tab:green"], strict=True
    ):
        axes[0].plot(time_ms, beats[i], color=color, linewidth=1, label=f"{label} real")
        axes[0].plot(
            time_ms, fitted[i], "--", color=color, linewidth=1.2, label=f"fit R²={r2[i]:.3f}"
        )
    axes[0].set_title(f"{name}: accepted fits")
    axes[0].legend(fontsize=7)

    axes[1].hist(np.clip(r2, 0, 1), bins=40, color="tab:blue")
    axes[1].axvline(min_r2, color="k", linestyle=":", label=f"min_r2 = {min_r2}")
    axes[1].set_title(f"R² of {len(r2)} fits ({int((~accepted).sum())} rejected)")
    axes[1].legend(fontsize=7)

    low, high = np.percentile(beats[accepted], [10, 90], axis=0)
    axes[2].fill_between(time_ms, low, high, color="tab:blue", alpha=0.2, label="real 10-90%")
    axes[2].plot(time_ms, samples.T, color="tab:red", linewidth=0.7, alpha=0.6)
    axes[2].plot([], [], color="tab:red", label="drawn from parameter sampler")
    axes[2].set_title(f"{name}: {len(samples)} beats drawn from the parameter sampler")
    axes[2].legend(fontsize=7)

    for ax in (axes[0], axes[2]):
        ax.set_xlabel("time from R-peak (ms)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/fitting.yaml", type=Path)
    args = parser.parse_args()

    config = load_config(ROOT / args.config)
    processed_dir = ROOT / config["processed_dir"]
    tables_dir = ROOT / config["tables_dir"]
    figures_dir = ROOT / config["figures_dir"]
    set_seed(config["seed"])
    device = get_device()

    space = ParamSpace(config["b_min"], math.radians(config["theta_r_max_deg"]))
    sim = EcgSyn().to(device)
    starts = torch.stack([*TYPICAL_PARAMS.values(), MCSHARRY_PARAMS]).to(device)
    generator = torch.Generator(device=device).manual_seed(config["seed"])

    per_beat: dict[str, list[np.ndarray]] = {}
    samplers: dict[str, ParamKDE] = {}
    quality_rows, summary_rows = [], []

    for name in config["classes"]:
        data = load_split(processed_dir, config["split"], classes=[name])
        beats = minmax_to_sim_range(data["X"])
        beats_t = torch.tensor(beats, device=device)

        mask = torch.ones(beats.shape[1], dtype=torch.bool, device=device)
        ignored = round(config["ignore_last_ms"].get(name, 0) / 1000 * SAMPLING_RATE)
        if ignored:
            mask[-ignored:] = False

        began = time.time()
        result = fit_beats(sim, beats_t, starts, space, mask, config["steps"], config["lr"])
        seconds = time.time() - began

        r2 = result.r2.cpu().numpy()
        accepted = r2 >= config["min_r2"]
        accepted_t = torch.as_tensor(accepted, device=device)
        sampler = ParamKDE.fit(
            result.raw[accepted_t], result.z0[accepted_t], space, config["jitter"]
        )
        samplers[name] = sampler

        with torch.no_grad():
            fitted = sim(result.params, result.z0).cpu().numpy()
            sample_params, sample_z0 = sampler.sample(N_SAMPLES_SHOWN, generator)
            samples = sim(sample_params, sample_z0).cpu().numpy()
        plot_class(
            name, beats, fitted, r2, accepted, config["min_r2"], samples,
            figures_dir / f"fits_{name}.png",
        )  # fmt: skip

        wins = np.bincount(result.start_index.cpu().numpy(), minlength=len(START_NAMES))
        quality_rows.append(
            [
                name,
                len(r2),
                int(accepted.sum()),
                int((~accepted).sum()),
                f"{np.median(r2):.4f}",
                f"{np.percentile(r2, 10):.4f}",
                f"{r2.min():.4f}",
                f"{seconds:.0f}",
                *wins.tolist(),
            ]
        )

        params = result.params[accepted_t].cpu().numpy()
        for w, wave in enumerate(WAVES):
            a, b, theta = params[:, 0, w], params[:, 1, w], np.degrees(params[:, 2, w])
            summary_rows.append(
                [name, wave]
                + [f"{v:.4f}" for v in (a.mean(), a.std(), b.mean(), b.std())]
                + [f"{v:.2f}" for v in (theta.mean(), theta.std())]
            )

        for key, value in {
            "class": np.full(len(r2), name),
            "record": data["record"],
            "r_peak": data["r_peak"],
            "params": result.params.cpu().numpy(),
            "raw": result.raw.cpu().numpy(),
            "z0": result.z0.cpu().numpy(),
            "r2": r2,
            "start_index": result.start_index.cpu().numpy(),
            "accepted": accepted,
        }.items():
            per_beat.setdefault(key, []).append(value)
        print(
            f"{name}: {len(r2)} beats in {seconds:.0f} s, median R² {np.median(r2):.3f}, "
            f"{int((~accepted).sum())} below {config['min_r2']}"
        )

    np.savez_compressed(
        processed_dir / "fitted_params.npz", **{k: np.concatenate(v) for k, v in per_beat.items()}
    )
    save_samplers(processed_dir / "param_samplers.npz", samplers)
    write_csv(
        tables_dir / "fit_quality.csv",
        ["class", "beats", "accepted", "rejected", "median_r2", "p10_r2", "min_r2", "seconds"]
        + [f"won_{s.replace(' ', '_')}" for s in START_NAMES],
        quality_rows,
    )
    write_csv(
        tables_dir / "fitted_param_summary.csv",
        ["class", "wave", "a_mean", "a_std", "b_mean", "b_std", "theta_deg_mean", "theta_deg_std"],
        summary_rows,
    )
    print(f"Saved parameter samplers to {processed_dir / 'param_samplers.npz'}")


if __name__ == "__main__":
    main()
