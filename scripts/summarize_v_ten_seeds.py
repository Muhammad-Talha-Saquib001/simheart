"""Summarise the V augmentation over 10 generator seeds per model (Phase 14).

Run from the repository root, after the Phase 14 steps of scripts/reproduce_results.py
(generator evaluation, augmentation and train-on-synthetic for V seeds 3-9):

    .venv/bin/python scripts/summarize_v_ten_seeds.py

Seeds 0-2 are the main study's rows, reused unchanged; seeds 3-9 add seven more generators
per model and seven more classifier seeds (classifier seed s uses generator seed s). Failed
generator runs stay in (the main rule); a version without the failed plain SimDCGAN runs is
descriptive. The patient bootstrap uses the main analysis's replicates (seed 0), checked by
reproducing the main control interval from seeds 0-2.

Primary comparisons, fixed before running (decision log §21): V at 1x over seeds 0-9,
SimDCGAN - DCGAN and SimDCGAN - control, on test AP. Everything else is descriptive.

Reads:  outputs/experiments/{results.jsonl, scores/*.npz}, data/processed/,
        outputs/tables/{bootstrap_setups.csv, failure_study.csv, generator_eval_per_seed.csv,
        generator_eval_warmup.csv, generator_eval_v_seeds3_9.csv}
Writes: outputs/tables/v_ten_seeds_{cells,comparisons,tstr,generators}.csv,
        outputs/figures/v_ten_seeds.png
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from simheart.classification.bootstrap import bootstrap_metrics, interval  # noqa: E402
from simheart.classification.datasets import real_binary  # noqa: E402
from simheart.classification.metrics import PAPER_RECALL  # noqa: E402
from simheart.reporting import mean_std, save_figure_atomic, write_csv_atomic  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "outputs" / "experiments"
TABLES, FIGURES = ROOT / "outputs" / "tables", ROOT / "outputs" / "figures"
BEAT_TYPE = "V"
SEEDS = list(range(10))
SETUPS = ["real", "oversample", "simulator", "dcgan", "simdcgan", "simdcgan_warmup"]
GANS = ["dcgan", "simdcgan", "simdcgan_warmup"]
AMOUNTS = [0.5, 1.0, 2.0]
COLORS = {
    "oversample": "tab:gray",
    "simulator": "tab:green",
    "dcgan": "tab:orange",
    "simdcgan": "tab:red",
    "simdcgan_warmup": "tab:purple",
}
# (name, minuend, subtrahend); "real" means the control at amount 0.
COMPARISONS = [
    ("simdcgan - dcgan", "simdcgan", "dcgan"),
    ("simdcgan - control", "simdcgan", "real"),
    ("dcgan - control", "dcgan", "real"),
    ("warm-up - dcgan", "simdcgan_warmup", "dcgan"),
    ("warm-up - control", "simdcgan_warmup", "real"),
    ("warm-up - simdcgan", "simdcgan_warmup", "simdcgan"),
]
PRIMARY = {(1.0, "simdcgan - dcgan"), (1.0, "simdcgan - control")}
GENERATOR_TABLES = [
    "generator_eval_per_seed.csv",
    "generator_eval_warmup.csv",
    "generator_eval_v_seeds3_9.csv",
]


def read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def cell_key(setup: str, amount: float) -> tuple[str, float]:
    return ("real", 0.0) if setup == "real" else (setup, amount)


def load_cells(results: list[dict], experiment: str = "augment") -> dict[tuple, list[dict]]:
    cells: dict[tuple, list[dict]] = {}
    for setup in SETUPS:
        for amount in [0.0] if setup == "real" else AMOUNTS:
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
                raise SystemExit(f"missing runs for {setup} {amount}x (have {len(rows)}/10)")
            cells[(setup, amount)] = rows
    return cells


def load_scores(rows: list[dict]) -> np.ndarray:
    return np.stack(
        [
            np.load(EXPERIMENTS / "scores" / f"{r['key'].replace('|', '_')}.npz")["test"]
            for r in rows
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--chunk", type=int, default=50, help="replicates per batch (memory)")
    args = parser.parse_args()

    results = [json.loads(x) for x in (EXPERIMENTS / "results.jsonl").read_text().splitlines()]
    cells = load_cells(results)
    control = {r["seed"]: r["ap_test"] for r in cells[("real", 0.0)]}
    failed = {
        int(r["seed"])
        for r in read_csv(TABLES / "failure_study.csv")
        if r["model"] == "simdcgan" and r["failed"] == "True"
    }
    ok_seeds = [s for s in SEEDS if s not in failed]

    cell_rows = []
    for (setup, amount), rows in cells.items():
        row = {"setup": setup, "amount": amount, "seeds": len(rows)}
        for metric in ("ap_test", "ap_test_strict"):
            row[metric], row[f"{metric}_sd"] = mean_std(r[metric] for r in rows)
        row["gain_vs_control"] = float(np.mean([r["ap_test"] - control[r["seed"]] for r in rows]))
        row["seeds_above_control"] = sum(r["ap_test"] > control[r["seed"]] for r in rows)
        row.update({f"ap_test_s{r['seed']}": r["ap_test"] for r in rows})
        cell_rows.append(row)

    _, labels, records = real_binary(ROOT / "data" / "processed", "test", BEAT_TYPE)
    recall = PAPER_RECALL[BEAT_TYPE]
    check_replicates(labels, records, cells, recall, args)

    scores = {cell: load_scores(rows) for cell, rows in cells.items()}
    # Descriptive: SimDCGAN and DCGAN on the seeds where plain SimDCGAN did not fail.
    for setup in ("simdcgan", "dcgan"):
        for amount in AMOUNTS:
            keep = [i for i, r in enumerate(cells[(setup, amount)]) if r["seed"] in ok_seeds]
            scores[(f"{setup}_ok", amount)] = scores[(setup, amount)][keep]
    print(f"Bootstrap: {len(scores)} cells x {args.n_boot} replicates ...", flush=True)
    points, samples = bootstrap_metrics(
        labels, scores, records, recall, args.n_boot, args.seed, chunk=args.chunk
    )

    comparisons = []
    for amount in AMOUNTS:
        for name, a, b in COMPARISONS:
            comparisons.append(
                compare(name, amount, cell_key(a, amount), cell_key(b, amount), cells, points,
                        samples, args.n_boot)
            )  # fmt: skip
        ok = compare(
            "simdcgan - dcgan (failed runs left out)",
            amount,
            ("simdcgan_ok", amount),
            ("dcgan_ok", amount),
            cells,
            points,
            samples,
            args.n_boot,
            seeds=ok_seeds,
        )
        comparisons.append(ok)

    tstr = []
    for method in ["simulator", *GANS]:
        rows = sorted(
            (
                r
                for r in results
                if r["experiment"] == "tstr"
                and r["class"] == BEAT_TYPE
                and r["setup"] == method
                and r["seed"] in SEEDS
            ),
            key=lambda r: r["seed"],
        )
        row = {"method": method, "seeds": len(rows)}
        row["ap_test"], row["ap_test_sd"] = mean_std(r["ap_test"] for r in rows)
        row.update({f"ap_test_s{r['seed']}": r["ap_test"] for r in rows})
        tstr.append(row)

    generators = generator_quality(cells, control, failed)

    write_csv_atomic(TABLES / "v_ten_seeds_cells.csv", cell_rows)
    write_csv_atomic(TABLES / "v_ten_seeds_comparisons.csv", comparisons)
    write_csv_atomic(TABLES / "v_ten_seeds_tstr.csv", tstr)
    write_csv_atomic(TABLES / "v_ten_seeds_generators.csv", generators["rows"])
    write_csv_atomic(TABLES / "v_ten_seeds_correlations.csv", generators["correlations"])
    plot(cell_rows, comparisons, generators["rows"])
    for c in comparisons:
        if c["amount"] == 1.0:
            print(
                f"1x {c['comparison']}: {c['diff']:+.3f} [{c['low']:+.3f}, {c['high']:+.3f}], "
                f"above 0 in {c['above_0']:.0%}, seeds {c['seed_wins']}"
                + (" (primary)" if c["primary"] else "")
            )
    print(f"Wrote {TABLES}/v_ten_seeds_*.csv and {FIGURES / 'v_ten_seeds.png'}")


def compare(name, amount, a, b, cells, points, samples, n_boot, seeds=None) -> dict:
    diff = interval(samples[a]["ap"] - samples[b]["ap"], points[a]["ap"] - points[b]["ap"])
    base_a = a if a in cells else (a[0].removesuffix("_ok"), a[1])
    base_b = b if b in cells else (b[0].removesuffix("_ok"), b[1])
    by_seed_b = {r["seed"]: r["ap_test"] for r in cells[base_b]}
    pairs = [
        (r["ap_test"], by_seed_b[r["seed"]])
        for r in cells[base_a]
        if seeds is None or r["seed"] in seeds
    ]
    return {
        "amount": amount,
        "comparison": name,
        "diff": diff["point"],
        "low": diff["low"],
        "high": diff["high"],
        "above_0": diff["above_0"],
        "usable": diff["usable"],
        "seed_wins": f"{sum(x > y for x, y in pairs)}/{len(pairs)}",
        "n_boot": n_boot,
        "primary": (amount, name) in PRIMARY,
    }


def check_replicates(labels, records, cells, recall, args) -> None:
    """The control from seeds 0-2 must reproduce the main bootstrap interval exactly."""
    main = next(
        r
        for r in read_csv(TABLES / "bootstrap_setups.csv")
        if r["class"] == BEAT_TYPE and r["setup"] == "real" and r["metric"] == "ap"
    )
    if int(main["n_boot"]) != args.n_boot:
        raise SystemExit(f"bootstrap_setups.csv used {main['n_boot']} replicates")
    control = {"c": load_scores(cells[("real", 0.0)][:3])}
    points, samples = bootstrap_metrics(
        labels, control, records, recall, args.n_boot, args.seed, chunk=args.chunk
    )
    own = interval(samples["c"]["ap"], points["c"]["ap"])
    for key, mine in [("value", own["point"]), ("low", own["low"]), ("high", own["high"])]:
        assert np.isclose(mine, float(main[key])), (key, mine, main[key])
    print("Bootstrap check: seeds 0-2 control reproduces bootstrap_setups.csv", flush=True)


def generator_quality(cells, control, failed) -> dict:
    """Each GAN generator's quality next to its own augmentation gain at 1x."""
    quality = {}
    for name in GENERATOR_TABLES:
        for r in read_csv(TABLES / name):
            if r["class"] == BEAT_TYPE and r["method"] in GANS:
                quality[(r["method"], int(r["seed"]))] = r
    rows = []
    for model in GANS:
        gains = {r["seed"]: r["ap_test"] - control[r["seed"]] for r in cells[(model, 1.0)]}
        for seed in SEEDS:
            q = quality[(model, seed)]
            rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "failed": model == "simdcgan" and seed in failed,
                    "mmd": float(q["mmd"]),
                    "diversity_ratio": float(q["diversity_ratio"]),
                    "euler_best_median": float(q["euler_best_median"]),
                    "gain_1x": gains[seed],
                }
            )
    correlations = []
    for subset, chosen in [
        ("all 30 generators", rows),
        ("working generators only", [r for r in rows if not r["failed"]]),
    ]:
        for measure in ("mmd", "euler_best_median"):
            rho, p = spearmanr([r[measure] for r in chosen], [r["gain_1x"] for r in chosen])
            correlations.append(
                {"subset": subset, "measure": measure, "n": len(chosen), "rho": rho, "p": p}
            )
    return {"rows": rows, "correlations": correlations}


def plot(cell_rows, comparisons, generators) -> None:
    fig, (left, middle, right) = plt.subplots(1, 3, figsize=(17, 4.6))
    control = next(r for r in cell_rows if r["setup"] == "real")
    left.axhspan(
        control["ap_test"] - control["ap_test_sd"],
        control["ap_test"] + control["ap_test_sd"],
        color="0.85",
        label="real only (± seed spread)",
    )
    for i, setup in enumerate(SETUPS[1:]):
        rows = [r for r in cell_rows if r["setup"] == setup]
        x = np.array([r["amount"] for r in rows]) * (1 + 0.03 * (i - 2))
        left.errorbar(x, [r["ap_test"] for r in rows], yerr=[r["ap_test_sd"] for r in rows],
                      marker="o", capsize=3, color=COLORS[setup], label=setup)  # fmt: skip
    left.set_xscale("log", base=2)
    left.set_xticks(AMOUNTS, [f"{a}×" for a in AMOUNTS])
    left.set_xlabel("synthetic V beats added (× real training V beats)")
    left.set_ylabel("test AP (DS2), mean ± spread over 10 seeds")
    left.set_title("V: augmentation, 10 generator seeds per model")
    left.legend(fontsize=7)

    shown = [c for c in comparisons if c["amount"] == 1.0]
    for y, c in enumerate(reversed(shown)):
        color = "tab:red" if c["primary"] else "tab:blue"
        middle.errorbar(c["diff"], y, xerr=[[c["diff"] - c["low"]], [c["high"] - c["diff"]]],
                        fmt="o", color=color, capsize=3)  # fmt: skip
    middle.set_yticks(range(len(shown)), [c["comparison"] for c in reversed(shown)], fontsize=7)
    middle.axvline(0, color="black", linewidth=0.8)
    middle.set_xlabel("difference in test AP at 1× (95 % interval over test patients)")
    middle.set_title("paired differences (red = primary)")

    for model in GANS:
        rows = [r for r in generators if r["model"] == model]
        right.scatter([r["mmd"] for r in rows], [r["gain_1x"] for r in rows],
                      color=COLORS[model], label=model, alpha=0.8)  # fmt: skip
    right.axhline(0, color="black", linewidth=0.8)
    right.set_xscale("log")
    right.set_xlabel("generator MMD to validation beats (log scale)")
    right.set_ylabel("AP gain over real only at 1× (same seed)")
    right.set_title("does generator quality predict augmentation gain?")
    right.legend(fontsize=7)
    fig.tight_layout()
    save_figure_atomic(fig, FIGURES / "v_ten_seeds.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
