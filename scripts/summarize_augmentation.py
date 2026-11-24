"""Summarise the augmentation experiments: tables and figures.

Run from the repository root, after scripts/run_augmentation.py:

    .venv/bin/python scripts/summarize_augmentation.py

Reads:  outputs/experiments/{results.jsonl, scores/}, data/processed/
Writes: outputs/tables/augmentation_results.csv      mean ± std over seeds, per setup/amount
        outputs/tables/augmentation_simdcgan_vs_dcgan.csv   per-seed wins on test AP
        outputs/tables/tstr_results.csv              (if train-on-synthetic was run)
        outputs/figures/augmentation_ap_<class>.png  test AP against amount, per setup
        outputs/figures/augmentation_pr_<class>.png  mean precision-recall curves at 1x
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve

from simheart.classification.datasets import real_binary
from simheart.reporting import MAIN_SEEDS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "experiments"
TABLES, FIGURES = ROOT / "outputs" / "tables", ROOT / "outputs" / "figures"
METRICS = ["ap_test", "p_at_paper_recall_test", "f1_test", "ap_test_strict", "f1_test_strict"]
SETUPS = ["real", "oversample", "simulator", "dcgan", "simdcgan"]
COLORS = {
    "real": "k",
    "oversample": "tab:gray",
    "simulator": "tab:green",
    "dcgan": "tab:orange",
    "simdcgan": "tab:red",
}


def cell(values: list[float]) -> str:
    return f"{np.mean(values):.3f} ± {np.std(values):.3f}"


def main() -> None:
    rows = [json.loads(line) for line in (OUT / "results.jsonl").read_text().splitlines()]
    # Main setups only; the supplementary warm-up setup has its own summary (Phase 13).
    augment = [
        r
        for r in rows
        if r["experiment"] == "augment" and r["setup"] in SETUPS and r["seed"] in MAIN_SEEDS
    ]
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in augment:
        groups[(r["class"], r["setup"], r["amount"])].append(r)
    classes = list(dict.fromkeys(r["class"] for r in augment))
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    with open(TABLES / "augmentation_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "setup", "amount", "seeds", *METRICS, "ap_gain_vs_control"])
        for (beat_type, setup, amount), group in sorted(groups.items()):
            control = {r["seed"]: r["ap_test"] for r in groups[(beat_type, "real", 0.0)]}
            gains = [r["ap_test"] - control[r["seed"]] for r in group if r["seed"] in control]
            writer.writerow(
                [beat_type, setup, amount, len(group)]
                + [cell([r[m] for r in group]) for m in METRICS]
                + [cell(gains) if gains else ""]
            )

    by_key = {(r["class"], r["setup"], r["amount"], r["seed"]): r for r in augment}
    with open(
        TABLES / "augmentation_simdcgan_vs_dcgan.csv", "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["class", "amount", "dcgan_ap", "simdcgan_ap", "seeds_simdcgan_better"])
        for (beat_type, setup, amount), group in sorted(groups.items()):
            if setup != "simdcgan":
                continue
            pairs = [
                (by_key[(beat_type, "dcgan", amount, r["seed"])]["ap_test"], r["ap_test"])
                for r in group
                if (beat_type, "dcgan", amount, r["seed"]) in by_key
            ]
            wins = sum(s > d for d, s in pairs)
            writer.writerow(
                [
                    beat_type,
                    amount,
                    cell([p[0] for p in pairs]),
                    cell([p[1] for p in pairs]),
                    f"{wins}/{len(pairs)}",
                ]
            )

    tstr = [r for r in rows if r["experiment"] == "tstr" and r["seed"] in MAIN_SEEDS]
    if tstr:
        with open(TABLES / "tstr_results.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["class", "method", "seeds", *METRICS])
            for beat_type in classes:
                for method in ["simulator", "dcgan", "simdcgan"]:
                    group = [r for r in tstr if r["class"] == beat_type and r["setup"] == method]
                    if group:
                        writer.writerow(
                            [beat_type, method, len(group)]
                            + [cell([r[m] for r in group]) for m in METRICS]
                        )

    for beat_type in classes:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        control = [r["ap_test"] for r in groups[(beat_type, "real", 0.0)]]
        ax.axhspan(
            np.mean(control) - np.std(control),
            np.mean(control) + np.std(control),
            color="k",
            alpha=0.1,
        )
        ax.axhline(np.mean(control), color="k", linestyle="--", label="real only (control)")
        for setup in SETUPS[1:]:
            amounts = sorted(a for (c, s, a) in groups if c == beat_type and s == setup)
            means = [
                np.mean([r["ap_test"] for r in groups[(beat_type, setup, a)]]) for a in amounts
            ]
            stds = [np.std([r["ap_test"] for r in groups[(beat_type, setup, a)]]) for a in amounts]
            ax.errorbar(
                amounts, means, yerr=stds, marker="o", capsize=3, label=setup, color=COLORS[setup]
            )
        ax.set_xlabel("added beats (× real training beats of the class)")
        ax.set_ylabel("test average precision (DS2)")
        ax.set_title(f"{beat_type}: detection on unseen patients (mean ± std over seeds)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(FIGURES / f"augmentation_ap_{beat_type}.png", dpi=110)
        plt.close(fig)

        _, y_test, _ = real_binary(ROOT / "data" / "processed", "test", beat_type)
        grid = np.linspace(0, 1, 101)
        fig, ax = plt.subplots(figsize=(6, 5))
        for setup in SETUPS:
            amount = 0.0 if setup == "real" else 1.0
            curves = []
            for r in groups.get((beat_type, setup, amount), []):
                path = OUT / "scores" / f"{r['key'].replace('|', '_')}.npz"
                precision, recall, _ = precision_recall_curve(y_test, np.load(path)["test"])
                curves.append(np.interp(grid, recall[::-1], precision[::-1]))
            if curves:
                ax.plot(
                    grid,
                    np.mean(curves, axis=0),
                    color=COLORS[setup],
                    label=f"{setup}" + ("" if setup == "real" else " (1×)"),
                )
        ax.set_xlabel("recall")
        ax.set_ylabel("precision")
        ax.set_title(f"{beat_type}: precision-recall on DS2 (mean over seeds)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(FIGURES / f"augmentation_pr_{beat_type}.png", dpi=110)
        plt.close(fig)

    print((TABLES / "augmentation_results.csv").read_text(encoding="utf-8"))
    print((TABLES / "augmentation_simdcgan_vs_dcgan.csv").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
