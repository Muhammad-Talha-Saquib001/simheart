"""Confidence intervals over test patients for the augmentation results (no retraining).

Run from the repository root, after scripts/run_augmentation.py:

    .venv/bin/python scripts/bootstrap_results.py            # 2,000 replicates
    .venv/bin/python scripts/bootstrap_results.py --n-boot 50   # quick check

Resamples the 22 DS2 test records with replacement (a patient-level bootstrap; see
simheart.classification.bootstrap) and rescores the saved test scores of every classifier.
Metrics are averaged over the classifier seeds inside each replicate, and all setups share
the same replicates, so differences are paired.

Primary comparisons, fixed before any interval was computed (decision log §16): V at 1x,
SimDCGAN - DCGAN and SimDCGAN - control, on test AP. Everything else is descriptive.

Reads:  outputs/experiments/{results.jsonl, scores/*.npz}, data/processed/
Writes: outputs/tables/bootstrap_setups.csv, outputs/tables/bootstrap_simdcgan_vs_dcgan.csv,
        outputs/figures/bootstrap_<class>.png
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from simheart.classification.bootstrap import bootstrap_metrics, interval  # noqa: E402
from simheart.classification.datasets import real_binary  # noqa: E402
from simheart.classification.metrics import PAPER_RECALL  # noqa: E402
from simheart.reporting import mean_std  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "outputs" / "experiments"
TABLES, FIGURES = ROOT / "outputs" / "tables", ROOT / "outputs" / "figures"
CLASSES = ["S", "V", "F"]
SETUPS = ["real", "oversample", "simulator", "dcgan", "simdcgan"]
METRICS = ["ap", "p_at_r"]
PRIMARY = {("V", "simdcgan", 1.0, "vs_dcgan"), ("V", "simdcgan", 1.0, "vs_control")}


def load_scores(beat_type: str, results: list[dict]) -> tuple[dict, dict]:
    """Test scores [n_seeds, N] and per-seed reported AP for each (setup, amount) cell."""
    cells: dict[tuple, list[dict]] = defaultdict(list)
    for r in results:
        if r["experiment"] == "augment" and r["class"] == beat_type:
            cells[(r["setup"], r["amount"])].append(r)
    scores, reported = {}, {}
    for cell, rows in sorted(cells.items(), key=lambda kv: (SETUPS.index(kv[0][0]), kv[0][1])):
        rows = sorted(rows, key=lambda r: r["seed"])
        files = [EXPERIMENTS / "scores" / f"{r['key'].replace('|', '_')}.npz" for r in rows]
        scores[cell] = np.stack([np.load(f)["test"] for f in files])
        reported[cell] = {
            "ap": [r["ap_test"] for r in rows],
            "p_at_r": [r["p_at_paper_recall_test"] for r in rows],
        }
    return scores, reported


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    results = [json.loads(x) for x in (EXPERIMENTS / "results.jsonl").read_text().splitlines()]
    processed = ROOT / "data" / "processed"
    setup_rows, pair_rows = [], []

    for beat_type in CLASSES:
        _, labels, records = real_binary(processed, "test", beat_type)
        scores, reported = load_scores(beat_type, results)
        points, samples = bootstrap_metrics(
            labels, scores, records, PAPER_RECALL[beat_type], args.n_boot, args.seed
        )
        control = ("real", 0.0)
        for cell in scores:
            setup, amount = cell
            for metric in METRICS:
                # The point value must reproduce the reported per-seed numbers.
                assert np.isclose(points[cell][metric], np.mean(reported[cell][metric]))
                own = interval(samples[cell][metric], points[cell][metric])
                diff = interval(
                    samples[cell][metric] - samples[control][metric],
                    points[cell][metric] - points[control][metric],
                )
                setup_rows.append(
                    {
                        "class": beat_type,
                        "setup": setup,
                        "amount": amount,
                        "metric": metric,
                        "value": own["point"],
                        "seed_sd": mean_std(reported[cell][metric])[1],
                        "low": own["low"],
                        "high": own["high"],
                        "diff_vs_control": diff["point"],
                        "diff_low": diff["low"],
                        "diff_high": diff["high"],
                        "diff_above_0": diff["above_0"],
                        "usable": own["usable"],
                        "n_boot": args.n_boot,
                        "primary": (beat_type, setup, amount, "vs_control") in PRIMARY
                        and metric == "ap",
                    }
                )
            if setup == "simdcgan" and ("dcgan", amount) in scores:
                for metric in METRICS:
                    other = ("dcgan", amount)
                    diff = interval(
                        samples[cell][metric] - samples[other][metric],
                        points[cell][metric] - points[other][metric],
                    )
                    pair_rows.append(
                        {
                            "class": beat_type,
                            "amount": amount,
                            "metric": metric,
                            "dcgan": points[other][metric],
                            "simdcgan": points[cell][metric],
                            "diff": diff["point"],
                            "low": diff["low"],
                            "high": diff["high"],
                            "above_0": diff["above_0"],
                            "usable": diff["usable"],
                            "n_boot": args.n_boot,
                            "primary": (beat_type, setup, amount, "vs_dcgan") in PRIMARY
                            and metric == "ap",
                        }
                    )
        plot_class(beat_type, setup_rows, pair_rows)
        print(f"{beat_type}: {len(scores)} cells, {args.n_boot} replicates", flush=True)

    TABLES.mkdir(parents=True, exist_ok=True)
    for name, rows in [
        ("bootstrap_setups", setup_rows),
        ("bootstrap_simdcgan_vs_dcgan", pair_rows),
    ]:
        with open(TABLES / f"{name}.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"Wrote {TABLES / 'bootstrap_setups.csv'} and bootstrap_simdcgan_vs_dcgan.csv")


def plot_class(beat_type: str, setup_rows: list[dict], pair_rows: list[dict]) -> None:
    """Forest plot: AP difference vs control per setup, and SimDCGAN - DCGAN, with 95 % CIs."""
    rows = [
        (f"{r['setup']} {r['amount']}× − control", r["diff_vs_control"], r["diff_low"],
         r["diff_high"], r["primary"])
        for r in setup_rows
        if r["class"] == beat_type and r["metric"] == "ap" and r["setup"] != "real"
    ] + [
        (f"SimDCGAN − DCGAN {r['amount']}×", r["diff"], r["low"], r["high"], r["primary"])
        for r in pair_rows
        if r["class"] == beat_type and r["metric"] == "ap"
    ]  # fmt: skip
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(rows) + 1.2))
    for y, (_label, point, low, high, primary) in enumerate(reversed(rows)):
        color = "tab:red" if primary else "tab:blue"
        ax.errorbar(point, y, xerr=[[point - low], [high - point]], fmt="o", color=color, capsize=3)
    ax.set_yticks(range(len(rows)), [r[0] for r in reversed(rows)], fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("difference in test AP (95 % interval over test patients)")
    ax.set_title(f"{beat_type}: paired differences (red = primary comparison)")
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"bootstrap_{beat_type}.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
