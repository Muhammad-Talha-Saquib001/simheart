"""Summarise the generator evaluation across seeds, and plot each source side by side.

Run from the repository root, after scripts/evaluate_generators.py:

    .venv/bin/python scripts/summarize_runs.py

Reads:  outputs/tables/generator_eval_per_seed.csv, data/synthetic/, data/processed/
Writes: outputs/tables/generator_comparison.csv          (mean ± std over seeds)
        outputs/tables/simdcgan_vs_dcgan.csv             (per-seed differences)
        outputs/figures/generators_<class>.png
"""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from simheart.data.dataset import load_split
from simheart.data.mitbih import SAMPLING_RATE
from simheart.data.scaling import minmax_to_sim_range
from simheart.data.synthetic import load_synthetic

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs" / "tables"
FIGURES = ROOT / "outputs" / "figures"
METRICS = [
    "mmd", "diversity_ratio", "euler_best_median", "hf_fraction_median",
    "nn_train_median", "near_copy_rate",
]  # fmt: skip
METHODS = ["real", "simulator", "dcgan", "simdcgan"]
TIME_MS = (np.arange(216) - 72) / SAMPLING_RATE * 1000


def better(metric: str, simdcgan: float, dcgan: float, real: dict[str, float]) -> bool:
    """Whether SimDCGAN's value is the better one for this metric."""
    if metric == "diversity_ratio":  # closest to real variety (1.0)
        return abs(simdcgan - 1) < abs(dcgan - 1)
    if metric == "hf_fraction_median":  # closest to the real share, on a log scale
        target = np.log(real[metric])
        return abs(np.log(simdcgan) - target) < abs(np.log(dcgan) - target)
    if metric == "nn_train_median":  # further from training beats = less copying
        return simdcgan > dcgan
    return simdcgan < dcgan  # mmd, euler_best_median, near_copy_rate: lower is better


def main() -> None:
    with open(TABLES / "generator_eval_per_seed.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    values: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        for metric in METRICS:
            values[(row["class"], row["method"])][metric].append(float(row[metric]))
    classes = list(dict.fromkeys(row["class"] for row in rows))

    with open(TABLES / "generator_comparison.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "method", "seeds", *METRICS])
        for beat_type in classes:
            for method in METHODS:
                v = values[(beat_type, method)]
                cells = [f"{np.mean(v[m]):.4f} ± {np.std(v[m]):.4f}" for m in METRICS]
                writer.writerow([beat_type, method, len(v["mmd"]), *cells])

    by_seed = {(r["class"], r["method"], r["seed"]): r for r in rows}
    with open(TABLES / "simdcgan_vs_dcgan.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "metric", "dcgan_mean", "simdcgan_mean", "seeds_simdcgan_better"])
        for beat_type in classes:
            real = {m: values[(beat_type, "real")][m][0] for m in METRICS}
            seeds = sorted({r["seed"] for r in rows if r["class"] == beat_type and r["seed"]})
            for metric in METRICS:
                pairs = [
                    (float(by_seed[(beat_type, "dcgan", s)][metric]),
                     float(by_seed[(beat_type, "simdcgan", s)][metric]))
                    for s in seeds
                ]  # fmt: skip
                wins = sum(better(metric, sd, d, real) for d, sd in pairs)
                writer.writerow(
                    [beat_type, metric]
                    + [f"{np.mean([p[i] for p in pairs]):.4f}" for i in (0, 1)]
                    + [f"{wins}/{len(pairs)}"]
                )

    FIGURES.mkdir(parents=True, exist_ok=True)
    for beat_type in classes:
        real = minmax_to_sim_range(load_split(ROOT / "data" / "processed", "validation",
                                              classes=[beat_type])["X"])  # fmt: skip
        low, high = np.percentile(real, [10, 90], axis=0)
        panels = [("real (validation)", real)]
        panels += [
            (method, load_synthetic(ROOT / "data" / "synthetic", method, beat_type, 0)[0])
            for method in ("simulator", "dcgan", "simdcgan")
        ]
        fig, axes = plt.subplots(1, 4, figsize=(20, 4), sharey=True)
        for ax, (name, beats) in zip(axes, panels, strict=True):
            ax.fill_between(TIME_MS, low, high, color="tab:blue", alpha=0.15, label="real 10-90%")
            ax.plot(TIME_MS, beats[:20].T, color="k" if name.startswith("real") else "tab:red",
                    linewidth=0.7, alpha=0.6)  # fmt: skip
            ax.set_title(f"{beat_type}: {name}" + ("" if name.startswith("real") else " (seed 0)"))
            ax.set_xlabel("time from R-peak (ms)")
        axes[0].legend(fontsize=8)
        axes[0].set_ylabel("scaled amplitude")
        fig.tight_layout()
        fig.savefig(FIGURES / f"generators_{beat_type}.png", dpi=110)
        plt.close(fig)

    print((TABLES / "generator_comparison.csv").read_text(encoding="utf-8"))
    print((TABLES / "simdcgan_vs_dcgan.csv").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
