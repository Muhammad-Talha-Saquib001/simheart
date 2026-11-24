"""Summarise the supplementary V augmentation with warm-up SimDCGAN generators (Phase 13).

Run from the repository root, after:

    .venv/bin/python scripts/evaluate_generators.py --classes V --models simdcgan_warmup \\
        --table outputs/tables/generator_eval_warmup.csv
    .venv/bin/python scripts/run_augmentation.py --classes V --setups simdcgan_warmup \\
        --settings outputs/experiments/pilot_choice.json
    .venv/bin/python scripts/run_augmentation.py --tstr --classes V --setups simdcgan_warmup \\
        --settings outputs/experiments/pilot_choice.json
    .venv/bin/python scripts/summarize_warmup_augmentation.py

The warm-up classifiers fine-tune the same base models with the same seeds as every main
setup, so all comparisons are paired. The patient bootstrap reuses the main analysis's
2,000 replicates (seed 0) and checks that the shared cells reproduce bootstrap_setups.csv.

Primary comparisons, fixed before running (decision log §20): V at 1x, warm-up - DCGAN and
warm-up - control, on test AP. Everything else is descriptive.

Reads:  outputs/experiments/{results.jsonl, scores/*.npz}, data/processed/,
        outputs/tables/{bootstrap_setups.csv, generator_eval_per_seed.csv,
        generator_eval_warmup.csv}
Writes: outputs/tables/warmup_augmentation.csv, warmup_comparisons.csv, warmup_tstr.csv,
        warmup_generators.csv; outputs/figures/warmup_augmentation.png
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
from simheart.reporting import mean_std  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "outputs" / "experiments"
TABLES, FIGURES = ROOT / "outputs" / "tables", ROOT / "outputs" / "figures"
BEAT_TYPE = "V"
WARMUP = "simdcgan_warmup"
SETUPS = ["real", "dcgan", "simdcgan", WARMUP]
COLORS = {"real": "black", "dcgan": "tab:orange", "simdcgan": "tab:red", WARMUP: "tab:purple"}
AMOUNTS = [0.5, 1.0, 2.0]
SEEDS = [0, 1, 2]
OTHERS = ["real", "dcgan", "simdcgan"]  # warm-up minus each of these
PRIMARY = {(1.0, "real"), (1.0, "dcgan")}
GENERATOR_METRICS = ["mmd", "diversity_ratio", "near_copy_rate", "euler_best_median",
                     "hf_fraction_median"]  # fmt: skip


def read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def cell_rows(results: list[dict], setup: str, amount: float) -> list[dict]:
    rows = [
        r
        for r in results
        if r["experiment"] == "augment"
        and r["class"] == BEAT_TYPE
        and r["setup"] == setup
        and r["amount"] == amount
        and r["seed"] in SEEDS
    ]
    rows = sorted(rows, key=lambda r: r["seed"])
    if [r["seed"] for r in rows] != SEEDS:
        raise SystemExit(f"missing runs for {setup} {amount}x: run run_augmentation.py first")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    results = [json.loads(x) for x in (EXPERIMENTS / "results.jsonl").read_text().splitlines()]
    cells = {("real", 0.0): cell_rows(results, "real", 0.0)}
    for setup in SETUPS[1:]:
        for amount in AMOUNTS:
            cells[(setup, amount)] = cell_rows(results, setup, amount)

    # Per-cell AP, with and without record 202.
    control = {r["seed"]: r["ap_test"] for r in cells[("real", 0.0)]}
    cell_table = []
    for (setup, amount), rows in cells.items():
        row = {"setup": setup, "amount": amount, "seeds": len(rows)}
        for metric in ("ap_test", "ap_test_strict"):
            row[metric], row[f"{metric}_sd"] = mean_std(r[metric] for r in rows)
        for r in rows:
            row[f"ap_test_s{r['seed']}"] = r["ap_test"]
        row["gain_vs_control"] = float(np.mean([r["ap_test"] - control[r["seed"]] for r in rows]))
        cell_table.append(row)

    # Patient bootstrap on the main analysis's replicates.
    _, labels, records = real_binary(ROOT / "data" / "processed", "test", BEAT_TYPE)
    scores = {
        cell: np.stack(
            [
                np.load(EXPERIMENTS / "scores" / f"{r['key'].replace('|', '_')}.npz")["test"]
                for r in rows
            ]
        )
        for cell, rows in cells.items()
    }
    points, samples = bootstrap_metrics(
        labels, scores, records, PAPER_RECALL[BEAT_TYPE], args.n_boot, args.seed
    )
    check_against_main(points, samples, args.n_boot)

    comparisons = []
    for amount in AMOUNTS:
        warm = (WARMUP, amount)
        for other in OTHERS:
            base = ("real", 0.0) if other == "real" else (other, amount)
            diff = interval(
                samples[warm]["ap"] - samples[base]["ap"], points[warm]["ap"] - points[base]["ap"]
            )
            pairs = zip(cells[warm], cells[base], strict=True)
            wins = sum(w["ap_test"] > b["ap_test"] for w, b in pairs)
            comparisons.append(
                {
                    "amount": amount,
                    "comparison": f"warmup - {other}",
                    "diff": diff["point"],
                    "low": diff["low"],
                    "high": diff["high"],
                    "above_0": diff["above_0"],
                    "usable": diff["usable"],
                    "seed_wins": f"{wins}/{len(SEEDS)}",
                    "n_boot": args.n_boot,
                    "primary": (amount, other) in PRIMARY,
                }
            )

    # Train-on-synthetic: synthetic positives only.
    tstr = []
    for method in ["simulator", "dcgan", "simdcgan", WARMUP]:
        rows = sorted(
            (
                r
                for r in results
                if r["experiment"] == "tstr" and r["class"] == BEAT_TYPE and r["setup"] == method
            ),
            key=lambda r: r["seed"],
        )
        if rows:
            row = {"method": method, **{f"ap_test_s{r['seed']}": r["ap_test"] for r in rows}}
            row["ap_test"], row["ap_test_sd"] = mean_std(r["ap_test"] for r in rows)
            tstr.append(row)

    # Generator quality of the seeds used here.
    generators = [
        {"method": r["method"], "seed": r["seed"], **{k: r[k] for k in GENERATOR_METRICS}}
        for path in ("generator_eval_per_seed.csv", "generator_eval_warmup.csv")
        for r in read_csv(TABLES / path)
        if r["class"] == BEAT_TYPE and r["method"] in ("dcgan", "simdcgan", WARMUP)
    ]

    TABLES.mkdir(parents=True, exist_ok=True)
    write_csv(TABLES / "warmup_augmentation.csv", cell_table)
    write_csv(TABLES / "warmup_comparisons.csv", comparisons)
    write_csv(TABLES / "warmup_tstr.csv", tstr)
    write_csv(TABLES / "warmup_generators.csv", generators)
    plot(cell_table, comparisons)
    for c in comparisons:
        print(
            f"{c['amount']}x {c['comparison']}: {c['diff']:+.3f} [{c['low']:+.3f}, "
            f"{c['high']:+.3f}], above 0 in {c['above_0']:.0%}, seeds {c['seed_wins']}"
            + (" (primary)" if c["primary"] else "")
        )
    print(f"Wrote {TABLES / 'warmup_augmentation.csv'} and {FIGURES / 'warmup_augmentation.png'}")


def check_against_main(points: dict, samples: dict, n_boot: int) -> None:
    """The shared cells must reproduce the main bootstrap exactly (same replicates)."""
    main_rows = read_csv(TABLES / "bootstrap_setups.csv")
    checked = 0
    for r in main_rows:
        cell = (r["setup"], float(r["amount"]))
        if r["class"] != BEAT_TYPE or r["metric"] != "ap" or cell not in points:
            continue
        if int(r["n_boot"]) != n_boot:
            raise SystemExit(f"bootstrap_setups.csv used {r['n_boot']} replicates, not {n_boot}")
        own = interval(samples[cell]["ap"], points[cell]["ap"])
        for key in ("value", "low", "high"):
            mine = own["point"] if key == "value" else own[key]
            assert np.isclose(mine, float(r[key])), (cell, key, mine, r[key])
        checked += 1
    assert checked == 1 + 2 * len(AMOUNTS), checked  # control, DCGAN and SimDCGAN cells
    print(f"Bootstrap check: {checked} shared cells match bootstrap_setups.csv", flush=True)


def plot(cell_table: list[dict], comparisons: list[dict]) -> None:
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.2))
    control = next(r for r in cell_table if r["setup"] == "real")
    left.axhspan(
        control["ap_test"] - control["ap_test_sd"],
        control["ap_test"] + control["ap_test_sd"],
        color="0.85",
        label="real only (± seed spread)",
    )
    for i, setup in enumerate(SETUPS[1:]):
        rows = [r for r in cell_table if r["setup"] == setup]
        x = np.array([r["amount"] for r in rows]) * (1 + 0.03 * (i - 1))
        left.errorbar(
            x,
            [r["ap_test"] for r in rows],
            yerr=[r["ap_test_sd"] for r in rows],
            marker="o",
            capsize=3,
            color=COLORS[setup],
            label=setup,
        )
    left.set_xscale("log", base=2)
    left.set_xticks(AMOUNTS, [f"{a}×" for a in AMOUNTS])
    left.set_xlabel("synthetic V beats added (× real training V beats)")
    left.set_ylabel("test AP (DS2), mean ± seed spread")
    left.set_title("V: augmentation with each generator")
    left.legend(fontsize=8)

    labels = [f"{c['comparison']} {c['amount']}×" for c in comparisons]
    for y, c in enumerate(reversed(comparisons)):
        color = "tab:red" if c["primary"] else "tab:blue"
        right.errorbar(
            c["diff"],
            y,
            xerr=[[c["diff"] - c["low"]], [c["high"] - c["diff"]]],
            fmt="o",
            color=color,
            capsize=3,
        )
    right.set_yticks(range(len(labels)), list(reversed(labels)), fontsize=8)
    right.axvline(0, color="black", linewidth=0.8)
    right.set_xlabel("difference in test AP (95 % interval over test patients)")
    right.set_title("warm-up SimDCGAN minus others (red = primary)")
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "warmup_augmentation.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
