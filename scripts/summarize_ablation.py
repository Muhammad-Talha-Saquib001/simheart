"""Summarise the Euler-loss ablation on V (Phase 15).

Run from the repository root, after the Phase 15 steps of scripts/reproduce_results.py
(training, generator evaluation, augmentation at 1x and train-on-synthetic):

    .venv/bin/python scripts/summarize_ablation.py

Every variant is SimDCGAN with the warm-up and one change (decision log §22). The reference
is SimDCGAN + warm-up itself (seeds 0-9, Phases 12 and 14). For each variant: the training
failure rate (Phase 12 rule), generator quality among successful runs, V augmentation at 1x
(all runs, failed ones included) and train-on-synthetic. All comparisons are descriptive.

Reads:  outputs/runs/<model>_s<seed>_V/summary.json, outputs/experiments/{results.jsonl,
        scores/*.npz}, data/processed/, outputs/tables/{generator_eval_warmup.csv,
        generator_eval_v_seeds3_9.csv, generator_eval_ablation.csv}
Writes: outputs/tables/ablation_{summary,comparisons}.csv, outputs/figures/ablation.png
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from simheart.classification.bootstrap import bootstrap_metrics, interval  # noqa: E402
from simheart.classification.datasets import real_binary  # noqa: E402
from simheart.classification.metrics import PAPER_RECALL  # noqa: E402
from simheart.reporting import (  # noqa: E402
    clopper_pearson,
    is_failed_run,
    mean_std,
    save_figure_atomic,
    write_csv_atomic,
)

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "outputs" / "experiments"
RUNS = ROOT / "outputs" / "runs"
TABLES, FIGURES = ROOT / "outputs" / "tables", ROOT / "outputs" / "figures"
BEAT_TYPE = "V"
SEEDS = list(range(10))
AMOUNT = 1.0
REFERENCE = "simdcgan_warmup"
VARIANTS = {
    REFERENCE: "reference (λ 0.0047, random pairing, KDE)",
    "abl_weight_low": "λ ¼× (0.0012)",
    "abl_weight_high": "λ 4× (0.019)",
    "abl_best_of_k": "best-of-16 pairing",
    "abl_gaussian": "paper's Gaussian sampler",
}
QUALITY = ["mmd", "diversity_ratio", "euler_best_median", "hf_fraction_median", "near_copy_rate"]
GENERATOR_TABLES = [
    "generator_eval_warmup.csv",
    "generator_eval_v_seeds3_9.csv",
    "generator_eval_ablation.csv",
]


def read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def rows_for(results: list[dict], experiment: str, setup: str, amount: float) -> list[dict]:
    rows = sorted(
        (
            r
            for r in results
            if r["experiment"] == experiment
            and r["class"] == BEAT_TYPE
            and r["setup"] == setup
            and r["amount"] == amount
            and r["seed"] in SEEDS
        ),
        key=lambda r: r["seed"],
    )
    if [r["seed"] for r in rows] != SEEDS:
        raise SystemExit(f"missing {experiment} runs for {setup} {amount}x ({len(rows)}/10)")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--chunk", type=int, default=50, help="replicates per batch (memory)")
    args = parser.parse_args()

    results = [json.loads(x) for x in (EXPERIMENTS / "results.jsonl").read_text().splitlines()]
    quality = {
        (r["method"], int(r["seed"])): r
        for name in GENERATOR_TABLES
        for r in read_csv(TABLES / name)
        if r["class"] == BEAT_TYPE and r["method"] in VARIANTS
    }
    control = rows_for(results, "augment", "real", 0.0)
    cells = {"control": control}
    summary = []
    for variant, label in VARIANTS.items():
        failed = [
            is_failed_run(
                json.loads((RUNS / f"{variant}_s{s}_{BEAT_TYPE}" / "summary.json").read_text())[
                    "best"
                ]
            )
            for s in SEEDS
        ]
        k, n = sum(failed), len(failed)
        low, high = clopper_pearson(k, n)
        row = {"variant": variant, "label": label, "runs": n, "failed": k, "ci_low": low,
               "ci_high": high}  # fmt: skip
        ok = [s for s, f in zip(SEEDS, failed, strict=True) if not f]
        for key in QUALITY:
            values = [float(quality[(variant, s)][key]) for s in ok]
            row[key], row[f"{key}_sd"] = mean_std(values) if values else (np.nan, np.nan)
        aug = rows_for(results, "augment", variant, AMOUNT)
        cells[variant] = aug
        row["ap_1x"], row["ap_1x_sd"] = mean_std(r["ap_test"] for r in aug)
        row["gain_vs_control"] = float(
            np.mean([a["ap_test"] - c["ap_test"] for a, c in zip(aug, control, strict=True)])
        )
        tstr = rows_for(results, "tstr", variant, 2.0)
        row["tstr_ap"], row["tstr_ap_sd"] = mean_std(r["ap_test"] for r in tstr)
        summary.append(row)

    _, labels, records = real_binary(ROOT / "data" / "processed", "test", BEAT_TYPE)
    scores = {
        name: np.stack(
            [
                np.load(EXPERIMENTS / "scores" / f"{r['key'].replace('|', '_')}.npz")["test"]
                for r in rows
            ]
        )
        for name, rows in cells.items()
    }
    print(f"Bootstrap: {len(scores)} cells x {args.n_boot} replicates ...", flush=True)
    points, samples = bootstrap_metrics(
        labels, scores, records, PAPER_RECALL[BEAT_TYPE], args.n_boot, args.seed, chunk=args.chunk
    )
    comparisons = []
    for variant in VARIANTS:
        for other in ["control"] if variant == REFERENCE else [REFERENCE, "control"]:
            diff = interval(
                samples[variant]["ap"] - samples[other]["ap"],
                points[variant]["ap"] - points[other]["ap"],
            )
            wins = sum(
                a["ap_test"] > b["ap_test"]
                for a, b in zip(cells[variant], cells[other], strict=True)
            )
            comparisons.append(
                {
                    "comparison": f"{variant} - {other}",
                    "amount": AMOUNT,
                    "diff": diff["point"],
                    "low": diff["low"],
                    "high": diff["high"],
                    "above_0": diff["above_0"],
                    "usable": diff["usable"],
                    "seed_wins": f"{wins}/{len(SEEDS)}",
                    "n_boot": args.n_boot,
                }
            )

    write_csv_atomic(TABLES / "ablation_summary.csv", summary)
    write_csv_atomic(TABLES / "ablation_comparisons.csv", comparisons)
    plot(summary, comparisons)
    for r in summary:
        print(
            f"{r['variant']}: failed {r['failed']}/{r['runs']}, MMD {r['mmd']:.3f}, "
            f"Euler {r['euler_best_median']:.2f}, AP 1x {r['ap_1x']:.3f} "
            f"(gain {r['gain_vs_control']:+.3f}), TSTR {r['tstr_ap']:.3f}"
        )
    print(f"Wrote {TABLES}/ablation_*.csv and {FIGURES / 'ablation.png'}")


def plot(summary: list[dict], comparisons: list[dict]) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.2))
    names = [r["variant"].removeprefix("abl_") for r in summary]
    x = np.arange(len(summary))
    panels = [
        ("failed", None, "failed runs (of 10)"),
        ("mmd", "mmd_sd", "MMD ↓ (successful runs)"),
        ("euler_best_median", "euler_best_median_sd", "Euler score ↓ (successful runs)"),
    ]
    for ax, (key, sd, title) in zip(axes[:3], panels, strict=True):
        values = [r[key] for r in summary]
        errors = [r[sd] for r in summary] if sd else None
        ax.bar(x, values, yerr=errors, capsize=3, color=["tab:purple"] + ["tab:blue"] * 4)
        ax.set_xticks(x, names, rotation=30, ha="right", fontsize=8)
        ax.set_title(title)
    shown = list(reversed(comparisons))
    for y, c in enumerate(shown):
        axes[3].errorbar(c["diff"], y, xerr=[[c["diff"] - c["low"]], [c["high"] - c["diff"]]],
                         fmt="o", color="tab:blue", capsize=3)  # fmt: skip
    axes[3].set_yticks(range(len(shown)), [c["comparison"] for c in shown], fontsize=7)
    axes[3].axvline(0, color="black", linewidth=0.8)
    axes[3].set_xlabel("difference in V test AP at 1× (95 % interval)")
    axes[3].set_title("augmentation (descriptive)")
    fig.tight_layout()
    save_figure_atomic(fig, FIGURES / "ablation.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
