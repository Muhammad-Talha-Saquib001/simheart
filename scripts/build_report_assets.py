"""Collect the results into a report package: Markdown tables, figures and a results draft.

Run from the repository root, at the end of scripts/reproduce_results.py:

    .venv/bin/python scripts/build_report_assets.py

All numbers come from the files the earlier steps wrote; nothing is retrained. Places that
need a human reading of the numbers are marked "Interpretation (to be written)".

Reads:  outputs/tables/*.csv, outputs/experiments/{results.jsonl, pilot_choice.json},
        exports/simheart-models/manifest.json, outputs/figures/*.png
Writes: outputs/report/results.md, outputs/report/tables/*.md, outputs/report/figures/*.png
"""

import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path

from simheart.reporting import markdown_table, mean_std

ROOT = Path(__file__).resolve().parents[1]
TABLES, FIGURES = ROOT / "outputs" / "tables", ROOT / "outputs" / "figures"
EXPERIMENTS = ROOT / "outputs" / "experiments"
REPORT = ROOT / "outputs" / "report"
CLASSES = ["S", "V", "F"]
METHODS = ["real", "simulator", "dcgan", "simdcgan"]
SETUPS = ["real", "oversample", "simulator", "dcgan", "simdcgan"]
TODO = "*Interpretation (to be written after reviewing the numbers).*"

# SimGANs paper, Table 1: precision at fixed recall, evaluated on the whole MIT-BIH database.
PAPER_TABLE_1 = {
    "S": {"recall": 0.41, "dcgan": 0.58, "simdcgan": 0.80},
    "V": {"recall": 0.91, "dcgan": 0.72, "simdcgan": 0.84},
    "F": {"recall": 0.60, "dcgan": 0.20, "simdcgan": 0.40},
}

DEVIATIONS = [
    "Python 3.14 in a project-local virtual environment (proposal: 3.11).",
    "Beat scaling uses the authors' *intended* range [-0.01563, 0.042557]; their code has a "
    "bug that maps to [-0.01563, 0.026927].",
    "Simulator time step 1/216 (one window = one heart cycle), as in the authors' code; the "
    "paper states 1/360.",
    "R wave placed at sample 72; angle differences wrapped to (-pi, pi]; baseline wander "
    "A = 0.005 (paper: 0.15 mV).",
    "Per-beat simulator parameters are fitted with constraints (positive widths, P-Q-R-S-T "
    "order) and multiple starting points; the last 50 ms of S windows are ignored.",
    "The per-class parameter distribution is a kernel density estimate over the fitted "
    "parameters (2 % jitter), not the paper's diagonal Gaussian, which gave realistic beats "
    "for only 6-17 % of samples.",
    "Generator with 7 transposed convolutions and a convolutional discriminator head (the "
    "authors' code; the paper's text says 6 layers and a fully connected head).",
    "Networks work on a [-1, 1] scale; the Euler loss maps back to simulator units.",
    "Euler-loss weight 0.0047 (equal to the authors' code); the paper's literal weight 1 "
    "collapsed training.",
    "Classifier input is 216 samples at 360 Hz (Kachuee et al.: 187 samples at 125 Hz).",
    "Three augmentation amounts (0.5x, 1x, 2x) instead of seven; evaluation on DS2 only.",
]


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fmt(mean: float, std: float, digits: int = 3) -> str:
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def generator_rows() -> list[dict]:
    per_seed = read_csv(TABLES / "generator_eval_per_seed.csv")
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in per_seed:
        groups[(row["class"], row["method"])].append(row)
    rows = []
    for c in CLASSES:
        for method in METHODS:
            group = groups.get((c, method), [])
            if not group:
                continue
            row = {"class": c, "method": method, "n": len(group)}
            for key, digits in [
                ("mmd", 3),
                ("diversity_ratio", 2),
                ("euler_best_median", 2),
                ("hf_fraction_median", 4),
                ("nn_train_median", 2),
                ("near_copy_rate", 3),
            ]:
                row[key] = fmt(*mean_std(float(r[key]) for r in group), digits)
            rows.append(row)
    return rows


def augmentation_rows(results: list[dict]) -> tuple[list[dict], dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in results:
        if r["experiment"] == "augment":
            groups[(r["class"], r["setup"], r["amount"])].append(r)
    rows = []
    for (c, setup, amount), group in sorted(
        groups.items(), key=lambda kv: (CLASSES.index(kv[0][0]), SETUPS.index(kv[0][1]), kv[0][2])
    ):
        control = {r["seed"]: r["ap_test"] for r in groups.get((c, "real", 0.0), [])}
        gains = [r["ap_test"] - control[r["seed"]] for r in group if r["seed"] in control]
        rows.append(
            {
                "class": c,
                "setup": setup,
                "amount": "—" if setup == "real" else f"{amount}×",
                "n": len(group),
                "ap": fmt(*mean_std(r["ap_test"] for r in group)),
                "gain": "—" if setup == "real" else fmt(*mean_std(gains)),
                "p_at_r": fmt(*mean_std(r["p_at_paper_recall_test"] for r in group)),
                "f1": fmt(*mean_std(r["f1_test"] for r in group)),
                "ap_strict": fmt(*mean_std(r["ap_test_strict"] for r in group)),
            }
        )
    return rows, groups


def paired_wins(groups: dict) -> list[dict]:
    rows = []
    for c in CLASSES:
        for amount in sorted({a for (cc, s, a) in groups if cc == c and s == "simdcgan"}):
            dc = {r["seed"]: r["ap_test"] for r in groups.get((c, "dcgan", amount), [])}
            sd = {r["seed"]: r["ap_test"] for r in groups.get((c, "simdcgan", amount), [])}
            seeds = sorted(set(dc) & set(sd))
            if seeds:
                wins = sum(sd[s] > dc[s] for s in seeds)
                rows.append(
                    {
                        "class": c,
                        "amount": f"{amount}×",
                        "dcgan": fmt(*mean_std(dc[s] for s in seeds)),
                        "simdcgan": fmt(*mean_std(sd[s] for s in seeds)),
                        "wins": f"{wins}/{len(seeds)}",
                    }
                )
    return rows


def main() -> None:
    (REPORT / "tables").mkdir(parents=True, exist_ok=True)
    (REPORT / "figures").mkdir(parents=True, exist_ok=True)
    results = []
    if (EXPERIMENTS / "results.jsonl").exists():
        results = [json.loads(x) for x in (EXPERIMENTS / "results.jsonl").read_text().splitlines()]
    sections: list[str] = [
        "# SimHeart — Results",
        "",
        "Generated by `scripts/build_report_assets.py` from the experiment "
        "outputs. Mean ± standard deviation over 3 seeds unless stated.",
        "",
    ]
    tables: dict[str, str] = {}

    split_rows = read_csv(TABLES / "class_counts_by_split.csv")
    tables["data_splits"] = markdown_table(
        split_rows,
        [
            ("split", "Split"),
            ("records", "Records"),
            ("N", "N"),
            ("S", "S"),
            ("V", "V"),
            ("F", "F"),
            ("total", "Total"),
        ],
    )
    fit_rows = read_csv(TABLES / "fit_quality.csv")
    tables["fit_quality"] = markdown_table(
        fit_rows,
        [
            ("class", "Class"),
            ("beats", "Beats"),
            ("rejected", "Rejected (R² < 0.8)"),
            ("median_r2", "Median R²"),
            ("p10_r2", "10th pct R²"),
        ],
    )
    gen_rows = generator_rows()
    tables["generator_fidelity"] = markdown_table(
        gen_rows,
        [
            ("class", "Class"),
            ("method", "Source"),
            ("mmd", "MMD to validation ↓"),
            ("diversity_ratio", "Diversity (1 = real)"),
            ("euler_best_median", "Physiology (Euler) ↓"),
            ("hf_fraction_median", "Power > 40 Hz"),
            ("nn_train_median", "Nearest-train distance ↑"),
            ("near_copy_rate", "Near-copy rate ↓"),
        ],
    )
    gen_wins = read_csv(TABLES / "simdcgan_vs_dcgan.csv")
    tables["generator_simdcgan_vs_dcgan"] = markdown_table(
        gen_wins,
        [
            ("class", "Class"),
            ("metric", "Measure"),
            ("dcgan_mean", "DCGAN"),
            ("simdcgan_mean", "SimDCGAN"),
            ("seeds_simdcgan_better", "Seeds where SimDCGAN better"),
        ],
    )

    aug_rows, groups = augmentation_rows(results)
    tables["augmentation"] = markdown_table(
        aug_rows,
        [
            ("class", "Class"),
            ("setup", "Setup"),
            ("amount", "Added"),
            ("ap", "Test AP"),
            ("gain", "AP gain vs control"),
            ("p_at_r", "Precision @ paper recall"),
            ("f1", "F1"),
            ("ap_strict", "Test AP (no record 202)"),
        ],
    )
    win_rows = paired_wins(groups)
    tables["augmentation_simdcgan_vs_dcgan"] = markdown_table(
        win_rows,
        [
            ("class", "Class"),
            ("amount", "Added"),
            ("dcgan", "DCGAN AP"),
            ("simdcgan", "SimDCGAN AP"),
            ("wins", "Seeds where SimDCGAN better"),
        ],
    )
    tstr = [r for r in results if r["experiment"] == "tstr"]
    tstr_rows = []
    for c in CLASSES:
        for method in ["simulator", "dcgan", "simdcgan"]:
            group = [r for r in tstr if r["class"] == c and r["setup"] == method]
            if group:
                tstr_rows.append(
                    {
                        "class": c,
                        "method": method,
                        "n": len(group),
                        "ap": fmt(*mean_std(r["ap_test"] for r in group)),
                        "f1": fmt(*mean_std(r["f1_test"] for r in group)),
                    }
                )
    tables["tstr"] = markdown_table(
        tstr_rows,
        [
            ("class", "Class"),
            ("method", "Synthetic positives from"),
            ("ap", "Test AP"),
            ("f1", "F1"),
        ],
    )

    paper_rows = []
    for c in CLASSES:
        ours = {
            s: [
                r["p_at_paper_recall_test"]
                for r in results
                if r["experiment"] == "augment"
                and r["class"] == c
                and r["setup"] == s
                and r["amount"] == 1.0
            ]
            for s in ("dcgan", "simdcgan")
        }
        if all(ours.values()):
            paper = PAPER_TABLE_1[c]
            dc_mean, sd_mean = mean_std(ours["dcgan"])[0], mean_std(ours["simdcgan"])[0]
            paper_rows.append(
                {
                    "class": c,
                    "recall": PAPER_TABLE_1[c]["recall"],
                    "paper": f"{paper['dcgan']:.2f} / {paper['simdcgan']:.2f}",
                    "ours": f"{dc_mean:.2f} / {sd_mean:.2f}",
                }
            )
    tables["paper_comparison"] = markdown_table(
        paper_rows,
        [
            ("class", "Class"),
            ("recall", "Recall"),
            ("paper", "Paper: DCGAN / SimDCGAN precision"),
            ("ours", "Ours (DS2, 1×): DCGAN / SimDCGAN precision"),
        ],
    )

    for name, table in tables.items():
        (REPORT / "tables" / f"{name}.md").write_text(table + "\n", encoding="utf-8")

    pilot = EXPERIMENTS / "pilot_choice.json"
    pilot_text = ""
    if pilot.exists():
        choice = json.loads(pilot.read_text())
        pilot_text = (
            f"Classifier settings chosen on validation only: learning rate "
            f"{choice['lr']}, {choice['base_epochs']} base epochs, "
            f"{choice['finetune_epochs']} fine-tune epochs (rule: {choice['rule']})."
        )
    manifest = ROOT / "exports" / "simheart-models" / "manifest.json"
    defaults_text = ""
    if manifest.exists():
        models = json.loads(manifest.read_text())["models"]
        defaults_text = (
            "Default UI generators (lowest validation MMD): "
            + ", ".join(
                f"{m['model']} {m['beat_type']} seed {m['seed']}" for m in models if m["is_default"]
            )
            + "."
        )

    sections += [
        "## 1. Data and splits",
        "",
        tables["data_splits"],
        "",
        "Patient-level split (de Chazal 2004): DS1 for training and validation (records 118, "
        "205, 223), DS2 for testing. Record 208 holds 90 % of the training F beats.",
        "",
        "## 2. Simulator fitting quality",
        "",
        tables["fit_quality"],
        "",
        "## 3. Generator fidelity and privacy",
        "",
        "2,000 beats per source, scored against real validation beats (other patients). The "
        "`real` row is the reference: real training beats for MMD, real validation beats for "
        "the distance measures (its near-copy rate is 5 % by construction).",
        "",
        tables["generator_fidelity"],
        "",
        "SimDCGAN vs DCGAN, per seed:",
        "",
        tables["generator_simdcgan_vs_dcgan"],
        "",
        TODO,
        "",
        "## 4. Augmentation: detection on unseen patients (DS2)",
        "",
        pilot_text,
        "",
        tables["augmentation"],
        "",
        "SimDCGAN vs DCGAN at the same amount, per seed:",
        "",
        tables["augmentation_simdcgan_vs_dcgan"],
        "",
        TODO,
        "",
        "## 5. Train on synthetic, test on real",
        "",
        tables["tstr"] or "*Not run.*",
        "",
        TODO,
        "",
        "## 6. Relation to the SimGANs paper",
        "",
        "The paper's Table 1 is evaluated on the **whole** MIT-BIH database (training patients "
        "included); ours is DS2 only (unseen patients), so the numbers are not directly "
        "comparable.",
        "",
        tables["paper_comparison"],
        "",
        "## 7. Deviations from the paper",
        "",
        *[f"- {d}" for d in DEVIATIONS],
        "",
        "## 8. Limitations",
        "",
        "- F beats come almost entirely from one training patient (record 208).",
        "- On the small classes (S, F) the discriminator overpowers the generator for both "
        "GANs; best-checkpoint selection keeps usable generators.",
        "- Single lead (MLII), 216-sample windows; S windows contain part of the next beat.",
        "- Three seeds: differences are reported with their spread, without significance tests.",
        "- The standard split shares one patient between training and test (201/202); "
        "results without record 202 are reported alongside.",
        "",
        "## 9. Hand-off",
        "",
        defaults_text,
        "",
    ]
    (REPORT / "results.md").write_text("\n".join(sections), encoding="utf-8")

    for pattern in [
        "examples_by_class.png",
        "fits_*.png",
        "generators_*.png",
        "augmentation_*.png",
        "api_demo.png",
    ]:
        for path in FIGURES.glob(pattern):
            shutil.copy2(path, REPORT / "figures" / path.name)
    print(f"Wrote {REPORT / 'results.md'} and {len(tables)} tables")


if __name__ == "__main__":
    main()
