"""Summarise the small-class GAN stabilisation (Phase 16): S and F, seeds 0-2.

Run from the repository root, after the Phase 16 steps of scripts/reproduce_results.py:

    .venv/bin/python scripts/summarize_small_class_fix.py

Each stabilised run (spectral normalisation in the discriminator plus one-sided label
smoothing, decision log §23) is paired by seed with the main run it modifies: dcgan_sn with
dcgan, simdcgan_sn with simdcgan. Measured: generator quality at the best checkpoint, and how
strongly the discriminator wins late in training (mean D(real) - D(generated) over the last
1,000 iterations). All comparisons are descriptive (3 seeds).

Reads:  outputs/runs/<model>_s<seed>_<class>/train_log.csv,
        outputs/tables/{generator_eval_per_seed.csv, generator_eval_sn.csv}
Writes: outputs/tables/small_class_fix.csv (one row per run),
        outputs/tables/small_class_fix_summary.csv, outputs/figures/small_class_fix.png
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from simheart.reporting import mean_std, save_figure_atomic, write_csv_atomic  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "outputs" / "runs"
TABLES, FIGURES = ROOT / "outputs" / "tables", ROOT / "outputs" / "figures"
CLASSES = ["S", "F"]
SEEDS = [0, 1, 2]
PAIRS = {"dcgan": "dcgan_sn", "simdcgan": "simdcgan_sn"}
QUALITY = ["mmd", "diversity_ratio", "euler_best_median", "hf_fraction_median"]
LATE = 4000  # iterations after this count as "late in training" (of 5,000)


def read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def d_advantage(model: str, seed: int, beat_type: str) -> float:
    """Mean D(real) - D(generated) over the last 1,000 iterations (1 = D always right)."""
    log = read_csv(RUNS / f"{model}_s{seed}_{beat_type}" / "train_log.csv")
    late = [float(r["d_real"]) - float(r["d_fake"]) for r in log if int(r["iteration"]) > LATE]
    return float(np.mean(late))


def main() -> None:
    quality = {
        (r["class"], r["method"], int(r["seed"])): r
        for name in ("generator_eval_per_seed.csv", "generator_eval_sn.csv")
        for r in read_csv(TABLES / name)
        if r["seed"] != ""
    }
    runs = []
    for beat_type in CLASSES:
        for plain, fixed in PAIRS.items():
            for model in (plain, fixed):
                for seed in SEEDS:
                    q = quality[(beat_type, model, seed)]
                    runs.append(
                        {
                            "class": beat_type,
                            "model": model,
                            "base": plain,
                            "stabilised": model == fixed,
                            "seed": seed,
                            **{k: float(q[k]) for k in QUALITY},
                            "d_advantage_late": d_advantage(model, seed, beat_type),
                        }
                    )

    summary = []
    for beat_type in CLASSES:
        for plain, fixed in PAIRS.items():
            before = [r for r in runs if r["class"] == beat_type and r["model"] == plain]
            after = [r for r in runs if r["class"] == beat_type and r["model"] == fixed]
            row = {"class": beat_type, "model": plain}
            for key in [*QUALITY, "d_advantage_late"]:
                row[f"{key}_plain"] = mean_std(r[key] for r in before)[0]
                row[f"{key}_stabilised"] = mean_std(r[key] for r in after)[0]
                diffs = [a[key] - b[key] for a, b in zip(after, before, strict=True)]
                row[f"{key}_diff"], row[f"{key}_diff_sd"] = mean_std(diffs)
                lower_better = key != "diversity_ratio"
                row[f"{key}_seeds_better"] = sum(
                    (d < 0) if lower_better else (d > 0) for d in diffs
                )
            summary.append(row)

    write_csv_atomic(TABLES / "small_class_fix.csv", runs)
    write_csv_atomic(TABLES / "small_class_fix_summary.csv", summary)
    plot(runs)
    for r in summary:
        print(
            f"{r['class']} {r['model']}: MMD {r['mmd_plain']:.3f} -> {r['mmd_stabilised']:.3f}, "
            f"Euler {r['euler_best_median_plain']:.2f} -> {r['euler_best_median_stabilised']:.2f}, "
            f"D advantage {r['d_advantage_late_plain']:.2f} -> "
            f"{r['d_advantage_late_stabilised']:.2f}"
        )
    print(f"Wrote {TABLES / 'small_class_fix_summary.csv'} and {FIGURES / 'small_class_fix.png'}")


def plot(runs: list[dict]) -> None:
    panels = [
        ("mmd", "MMD to validation ↓"),
        ("euler_best_median", "Euler score ↓"),
        ("hf_fraction_median", "power above 40 Hz ↓"),
        ("d_advantage_late", "D(real) − D(generated), last 1,000 iterations"),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(18, 4))
    groups = [(c, m) for c in CLASSES for m in PAIRS]
    for ax, (key, title) in zip(axes, panels, strict=True):
        for i, (beat_type, plain) in enumerate(groups):
            for j, model in enumerate((plain, PAIRS[plain])):
                values = [r[key] for r in runs if r["class"] == beat_type and r["model"] == model]
                x = i + (j - 0.5) * 0.35
                ax.scatter([x] * len(values), values, color=["0.5", "tab:blue"][j], s=18,
                           label=["main", "stabilised"][j] if i == 0 else None)  # fmt: skip
        ax.set_xticks(range(len(groups)), [f"{c}\n{m}" for c, m in groups], fontsize=8)
        ax.set_title(title, fontsize=9)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    save_figure_atomic(fig, FIGURES / "small_class_fix.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
