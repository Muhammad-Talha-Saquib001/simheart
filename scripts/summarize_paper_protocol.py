"""Summarise the paper-protocol replication (run_augmentation.py --whole-database).

Run from the repository root:

    .venv/bin/python scripts/summarize_paper_protocol.py

The SimGANs paper's Table 1 trains on DS1 and evaluates on the whole MIT-BIH database. This
script puts our numbers in that setting next to the paper's, split into the parts of the
database: training patients (seen), validation patients and DS2 (unseen). It also checks
that the repeated fine-tuning reproduced the original DS2 results (rule: every AP within
0.005; decision log §17).

Reads:  outputs/experiments/paper_protocol/results.jsonl, outputs/experiments/results.jsonl
Writes: outputs/tables/paper_protocol.csv, outputs/tables/paper_protocol_reproducibility.json,
        outputs/figures/paper_protocol.png
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from simheart.reporting import PAPER_TABLE_1, mean_std  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "outputs" / "experiments"
TABLES, FIGURES = ROOT / "outputs" / "tables", ROOT / "outputs" / "figures"
CLASSES = ["S", "V", "F"]
SETUPS = ["real", "oversample", "simulator", "dcgan", "simdcgan"]
PARTS = ["whole", "seen", "validation", "unseen"]
METRICS = {"p_at_paper_recall": "precision at the paper's recall", "ap": "AP"}
TOLERANCE = 0.005


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def reproducibility(new: list[dict], old: list[dict]) -> dict:
    """Largest difference in DS2 AP between the repeated and the original fine-tunings."""
    original = {r["key"]: r["ap_test"] for r in old}
    diffs = {r["key"]: abs(r["ap_test"] - original[r["key"]]) for r in new if r["key"] in original}
    by_class = defaultdict(list)
    for key, d in diffs.items():
        by_class[key.split("|")[1]].append(d)
    return {
        "models_compared": len(diffs),
        "identical": int(sum(d == 0 for d in diffs.values())),
        "max_abs_diff": float(max(diffs.values())),
        "max_abs_diff_by_class": {c: float(max(v)) for c, v in sorted(by_class.items())},
        "differing_models": sorted(k for k, d in diffs.items() if d > 0),
        "tolerance": TOLERANCE,
        "passed": bool(max(diffs.values()) <= TOLERANCE),
    }


def main() -> None:
    new = read_jsonl(EXPERIMENTS / "paper_protocol" / "results.jsonl")
    check = reproducibility(new, read_jsonl(EXPERIMENTS / "results.jsonl"))
    print(
        f"Reproducibility: {check['identical']}/{check['models_compared']} identical, "
        f"largest DS2 AP difference {check['max_abs_diff']:.4f} "
        f"({'within' if check['passed'] else 'OUTSIDE'} {TOLERANCE})"
    )

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in new:
        if r["experiment"] == "augment":
            groups[(r["class"], r["setup"], r["amount"])].append(r)
    rows = []
    for (c, setup, amount), group in sorted(
        groups.items(), key=lambda kv: (CLASSES.index(kv[0][0]), SETUPS.index(kv[0][1]), kv[0][2])
    ):
        row = {"class": c, "setup": setup, "amount": amount, "seeds": len(group)}
        for metric in METRICS:
            for part in PARTS:
                mean, sd = mean_std(r[f"{metric}_{part}"] for r in group)
                row[f"{metric}_{part}"], row[f"{metric}_{part}_sd"] = mean, sd
        row["paper_p_at_paper_recall"] = PAPER_TABLE_1[c].get(setup, "")
        rows.append(row)

    TABLES.mkdir(parents=True, exist_ok=True)
    with open(TABLES / "paper_protocol.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (TABLES / "paper_protocol_reproducibility.json").write_text(json.dumps(check, indent=2))
    plot(rows)
    print(f"Wrote {TABLES / 'paper_protocol.csv'} and {FIGURES / 'paper_protocol.png'}")


def plot(rows: list[dict]) -> None:
    """Per class at 1x: the paper's precision against ours on each part of the database."""
    groups = ["paper\n(whole DB)", "ours\nwhole DB", "ours\nseen", "ours\nvalidation", "ours\nDS2"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for ax, c in zip(axes, CLASSES, strict=True):
        cell = {
            r["setup"]: r
            for r in rows
            if r["class"] == c and (r["amount"] == 1.0 or r["setup"] == "real")
        }
        x = np.arange(len(groups))
        for offset, setup, color in [(-0.2, "dcgan", "tab:orange"), (0.2, "simdcgan", "tab:red")]:
            ours = [cell[setup][f"p_at_paper_recall_{p}"] for p in PARTS]
            sds = [np.nan] + [cell[setup][f"p_at_paper_recall_{p}_sd"] for p in PARTS]
            values = [PAPER_TABLE_1[c][setup], *ours]
            ax.bar(x + offset, values, 0.4, yerr=sds, capsize=2, color=color, label=setup)
        control = [cell["real"][f"p_at_paper_recall_{p}"] for p in PARTS]
        ax.scatter(x[1:], control, marker="_", s=400, color="black", label="real only", zorder=3)
        ax.set_xticks(x, groups, fontsize=8)
        ax.set_title(f"{c}: precision at recall {PAPER_TABLE_1[c]['recall']} (1× added)")
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("precision")
    axes[0].legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "paper_protocol.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
