"""Collect the results into a report package: Markdown tables, figures and a results draft.

Run from the repository root, at the end of scripts/reproduce_results.py:

    .venv/bin/python scripts/build_report_assets.py

All numbers come from the files the earlier steps wrote; nothing is retrained. Places that
need a human reading of the numbers come from INTERPRETATION, written after reviewing them.

Reads:  outputs/tables/*.csv, outputs/experiments/{results.jsonl, pilot_choice.json},
        exports/simheart-models/manifest.json, outputs/figures/*.png
Writes: outputs/report/results.md, outputs/report/tables/*.md, outputs/report/figures/*.png
"""

import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path

from simheart.reporting import PAPER_TABLE_1, markdown_table, mean_std

ROOT = Path(__file__).resolve().parents[1]
TABLES, FIGURES = ROOT / "outputs" / "tables", ROOT / "outputs" / "figures"
EXPERIMENTS = ROOT / "outputs" / "experiments"
REPORT = ROOT / "outputs" / "report"
CLASSES = ["S", "V", "F"]
METHODS = ["real", "simulator", "dcgan", "simdcgan"]
SETUPS = ["real", "oversample", "simulator", "dcgan", "simdcgan"]

# Runs that failed during training, found in the final generator evaluation. The main tables
# keep them (all pre-registered seeds are reported); a clearly labelled supplementary table
# repeats the affected rows without them. Key: (class, method, seed).
FAILED_RUNS = {
    ("V", "simdcgan", 2): "broke down by iteration 500 and never recovered: near-identical, "
    "noise-like beats (diversity ratio 0.008, 90 % of power above 40 Hz, MMD 0.70)",
}


# Reading of the numbers, written by hand after the full run of 23 November 2026. If the
# experiments are rerun, check these statements against the new tables.
INTERPRETATION = {
    "generators": (
        "**Interpretation.** All generators are close to the real-data reference on realism "
        "and variety; differences between them are small next to the gap between training and "
        "validation patients. On the small classes the Euler loss does what it is meant to "
        "do: SimDCGAN beats are smoother than DCGAN beats in every seed (power above 40 Hz: "
        "S 0.013 vs 0.015, F 0.004 vs 0.010, 3/3 seeds each) and obey the heart model better "
        "(Euler score F 7.4 vs 18.0, 3/3 seeds; S 14.6 vs 15.7, 2/3). The cost is slightly "
        "higher MMD on S and F. The V means are dominated by one failed run (SimDCGAN seed 2, "
        "see 4b); the other two V SimDCGAN seeds have the lowest MMD of any V generator. "
        "**Privacy:** no source produces near-copies of training beats (near-copy rate 0 % "
        "for both GANs, at most 0.4 % for the simulator). Generated beats lie closer to the "
        "training beats (median distance 1.0-1.2) than other patients' real beats do "
        "(1.6-2.7), which is expected from a model of those patients and is not memorisation."
    ),
    "augmentation": (
        "**Interpretation.** *V (the only class the classifier can detect on new patients):* "
        "the control already reaches AP 0.90. SimDCGAN is the only setup whose mean AP is "
        "above the control at every amount, and it beats DCGAN at the same amount in all 9 "
        "paired comparisons (3 seeds x 3 amounts). Its gain over the control is small "
        "(+0.005 to +0.013 AP) and within the seed spread (and the patient-level intervals of "
        "section 4c), so the clearer effect is that "
        "DCGAN beats (-0.02 to -0.03) and simple oversampling (-0.01 to -0.04) *hurt*, "
        "while SimDCGAN beats do not. At the paper's operating point (recall 0.91), "
        "precision rises from 0.50 (control) to "
        "0.61-0.65 with SimDCGAN, against 0.41-0.49 with DCGAN, but with large spread "
        "(± 0.06-0.16). The 9 paired wins come from only 3 independent seeds (3/3 per amount), "
        "so they are consistent evidence, not a significance test. "
        "*S and F:* every setup stays at or near chance (S prevalence in DS2 3.7 %, AP about "
        "0.04-0.08; F prevalence 0.8 %, AP 0.006-0.007). Augmentation cannot fix this. S "
        "beats are recognised mainly by their early timing, which a single-beat classifier "
        "cannot see; F training beats come 90 % from one patient (record 208: 372 of 414) and "
        "F test beats 93 % from another (record 213: 362 of 388), so no beat-shape model "
        "learned from 208 transfers. Removing record 202 changes nothing."
    ),
    "sensitivity": (
        "**Interpretation.** Without the failed run the picture for V is the same: SimDCGAN "
        "has the lowest MMD of all V sources (0.115 vs 0.124 for DCGAN) and still beats "
        "DCGAN on every remaining paired seed (2/2 at each amount). Even the failed run's "
        "augmentation did not hurt (seed 2 AP 0.91-0.92 vs control 0.89): its noise-like "
        "beats are easy to tell apart from real beats, so the classifier effectively ignores "
        "them. The failure shows up clearly only when the synthetic beats are all the "
        "classifier sees (section 5)."
    ),
    "reliability": (
        "**Interpretation.** *Without a warm-up, SimDCGAN is unreliable on V:* 6 of 10 runs "
        "failed (95 % interval 26-88 %), against 0 of 10 for DCGAN and 0 of 10 for SimDCGAN "
        "with the warm-up (each 0-31 %). Fisher's test gives p = 0.011 for both comparisons; "
        "they were planned in advance but are descriptive, and the intervals do not overlap. "
        "*How the runs fail:* in 9 of the 10 plain SimDCGAN runs the first checkpoint "
        "(iteration 250) is already in the failure region. Three of them escaped later "
        "(seed 0 by iteration 1,500, seeds 1 and 6 only by 3,250-3,500), six never did, and "
        "only seed 3 trained normally from the start. DCGAN and the warm-up runs spend at most "
        "2 and 4 of 20 checkpoints in the failure region, all early. So the full Euler weight "
        "from the first iteration breaks training, and whether a run recovers within 5,000 "
        "iterations is luck; with longer training some failed runs might recover (not "
        "tested). *The warm-up costs nothing measurable:* among successful runs, warm-up "
        "MMD is 0.099 ± 0.014 against 0.102 ± 0.005 without it, and the Euler score 7.5 ± "
        "0.7 against 7.8 ± 1.1 (paired on the 4 seeds where both succeeded: MMD -0.001 ± "
        "0.008, Euler -0.12 ± 0.86). Both SimDCGAN variants keep a lower Euler score than "
        "DCGAN (9.9 ± 4.9, or 8.3 without DCGAN seed 4 at 24.0) and less power above 40 Hz. "
        "*What this means for the main results:* the augmentation experiments use SimDCGAN "
        "seeds 0-2 without warm-up; one failed and the other two recovered only after 1,500 and "
        "3,250 iterations, so 1 failure "
        "in 3 was, if anything, a lucky draw. The augmentation was not rerun with warm-up "
        "generators. The finding for the report: the Euler loss needs a warm-up to train "
        "reliably on V; with one, it trains as reliably as DCGAN."
    ),
    "bootstrap": (
        "**Interpretation.** *Primary comparisons (V, 1x):* SimDCGAN beats DCGAN by +0.036 "
        "AP, with a 95 % interval of [-0.001, +0.090]; the difference is above 0 in 97 % of "
        "resamples. That is strong but not conclusive evidence: the interval just reaches 0. "
        "SimDCGAN against the control gives +0.009 [-0.007, +0.055] (above 0 in 74 %), so "
        "there is **no evidence that SimDCGAN augmentation improves on real data alone**. "
        "The honest summary is that SimDCGAN beats are *safer* to add than DCGAN beats "
        "(DCGAN is below the control in 89-93 % of resamples, oversampling in 85-96 %), not "
        "that they make detection better. "
        "*Which patients matter far more than which seed:* the control's V AP ranges over "
        "[0.71, 0.97] across resampled test sets, against ± 0.006 across seeds, so the "
        "± values in section 4 understate the real uncertainty about ten-fold (an interval "
        "that wide corresponds to a spread of about ± 0.066). "
        "*Descriptive results:* at 2x the SimDCGAN - DCGAN interval excludes 0 (+0.043 "
        "[+0.002, +0.096]), and precision at recall 0.91 is higher by +0.14 to +0.20, with "
        "intervals that just reach 0. On S and F every interval is wide or sits on 0; the "
        "two that exclude 0 (S DCGAN 2x below the control; an F difference of 0.001) are "
        "what about 70 intervals produce by chance and aren't interpreted."
    ),
    "tstr": (
        "**Interpretation.** Trained on synthetic V beats alone (real non-V beats), the "
        "classifier still finds real V beats in new patients: the two working SimDCGAN seeds "
        "give the best results (AP 0.91 and 0.93, against 0.90, 0.87 and 0.73 for DCGAN and "
        "0.52-0.88 for the simulator). The failed seed gives AP 0.08, which is why the "
        "SimDCGAN mean is 0.64 ± 0.40. So, when training succeeds, SimDCGAN beats carry the "
        "most useful V information, but SimDCGAN training is less reliable than DCGAN "
        "training (1 failure in 9 SimDCGAN runs, 0 in 9 DCGAN runs). S and F are at chance, "
        "as in section 4."
    ),
    "paper": (
        "**Interpretation.** *Primary question (V, 1x, the paper's protocol):* SimDCGAN is "
        "above DCGAN on the whole database (0.63 ± 0.04 vs 0.59 ± 0.02), so the paper's "
        "ordering reproduces, but the gap is smaller (0.04 vs the paper's 0.12) and both "
        "values are below the paper's (0.72 / 0.84). The split by patient group explains why "
        "the protocol matters so much: on the patients the classifier was trained on, "
        "precision is 0.97-0.98 for V and F and 0.82-0.94 for S, *above* the paper's numbers, "
        "and every setup looks alike (a ceiling). On patients it never trained on, precision "
        "drops to 0.41-0.65 (V, DS2), 0.12-0.17 (V, validation, dominated by record 118) and "
        "near chance for S and F. A whole-database number is therefore mostly a statement "
        "about the *mix* of seen and unseen beats, not about the generators. "
        "*S and F:* even under the paper's protocol we are far below its values (S 0.05-0.09 "
        "vs 0.58-0.80; F 0.01-0.02 vs 0.20-0.40). About 72 % of all S beats and half of all F "
        "beats come from patients outside our training set. Reaching the paper's recall "
        "level means finding some of those, and on them we are at chance, so precision "
        "collapses. "
        "The paper's S and F numbers lie between our seen and unseen results, which would "
        "fit an evaluation in which most beats come from patients seen during training (for "
        "example a beat-level split); the paper doesn't give enough detail to check this, so "
        "it is a hypothesis, not a finding. What we can say: with a patient-level training "
        "set, the paper's S and F results do not reproduce, and its V ordering does, with a "
        "smaller margin."
    ),
    "reproducibility": (
        "**Why the reproducibility rule was exceeded (investigated before writing up).** "
        "All 9 base models and all V and F models were reproduced exactly; the 12 differing "
        "models are all S models at 2x. Repeating one of them (S, SimDCGAN, 2x, seed 1) in "
        "five fresh processes gave AP 0.0402 four times and 0.0497 once (original run: "
        "0.0342); inside one process it never varies. The cause is cuDNN's kernel tuning "
        "(decision log §13): it times its kernels in each new process, and timing noise "
        "occasionally selects a different, equally deterministic kernel whose rounding "
        "differs; at chance level (S) such tiny differences reorder the few positives. With "
        "cuDNN switched off the run is fully repeatable but about 25 times slower. The "
        "variation (up to about 0.016 AP, on S only) is far smaller than the patient-level "
        "intervals for S (section 4c) and changes no conclusion; numbers in this report can "
        "differ in that range when the pipeline is rerun."
    ),
}


def is_failed(beat_type: str, method: str, seed) -> bool:
    return seed not in ("", None) and (beat_type, method, int(seed)) in FAILED_RUNS


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


def generator_rows(drop_failed: bool = False) -> list[dict]:
    per_seed = read_csv(TABLES / "generator_eval_per_seed.csv")
    if drop_failed:
        per_seed = [r for r in per_seed if not is_failed(r["class"], r["method"], r["seed"])]
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


def ci(row: dict, point: str, low: str, high: str) -> str:
    """'0.009 [-0.012, 0.031]': value and 95 % interval over test patients."""
    return f"{float(row[point]):+.3f} [{float(row[low]):+.3f}, {float(row[high]):+.3f}]"


def bootstrap_tables() -> dict[str, list[dict]]:
    """Rows for the confidence-interval tables (empty when the bootstrap hasn't run)."""
    setups = read_csv(TABLES / "bootstrap_setups.csv")
    pairs = read_csv(TABLES / "bootstrap_simdcgan_vs_dcgan.csv")
    primary = [
        {
            "comparison": f"{r['class']} {float(r['amount'])}×: SimDCGAN − "
            + ("DCGAN" if "dcgan" in r else "control"),
            "diff": ci(r, "diff", "low", "high")
            if "dcgan" in r
            else ci(r, "diff_vs_control", "diff_low", "diff_high"),
            "above": f"{float(r['above_0' if 'dcgan' in r else 'diff_above_0']):.0%}",
            "usable": r["usable"],
        }
        for r in pairs + setups
        if r["primary"] == "True"
    ]
    vs_control = [
        {
            "class": r["class"],
            "setup": r["setup"],
            "amount": f"{float(r['amount'])}×",
            "value": f"{float(r['value']):.3f} [{float(r['low']):.3f}, {float(r['high']):.3f}]",
            "diff": ci(r, "diff_vs_control", "diff_low", "diff_high"),
            "above": f"{float(r['diff_above_0']):.0%}",
            "usable": r["usable"],
        }
        for r in setups
        if r["metric"] == "ap" and r["setup"] != "real"
    ]
    control = [
        {
            "class": r["class"],
            "value": f"{float(r['value']):.3f} [{float(r['low']):.3f}, {float(r['high']):.3f}]",
            "seed_sd": f"± {float(r['seed_sd']):.3f}",
            "usable": r["usable"],
        }
        for r in setups
        if r["metric"] == "ap" and r["setup"] == "real"
    ]
    vs_dcgan = [
        {
            "class": r["class"],
            "amount": f"{float(r['amount'])}×",
            "metric": "AP" if r["metric"] == "ap" else "Precision @ paper recall",
            "diff": ci(r, "diff", "low", "high"),
            "above": f"{float(r['above_0']):.0%}",
            "usable": r["usable"],
        }
        for r in pairs
    ]
    return {"primary": primary, "control": control, "vs_control": vs_control, "vs_dcgan": vs_dcgan}


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

    # Supplementary (not pre-registered): the affected classes without the failed runs.
    failed_classes = sorted({c for c, _, _ in FAILED_RUNS}, key=CLASSES.index)
    kept = [r for r in results if not is_failed(r["class"], r["setup"], r["seed"])]
    sens_gen = [r for r in generator_rows(drop_failed=True) if r["class"] in failed_classes]
    sens_aug, sens_groups = augmentation_rows(kept)
    sens_aug = [r for r in sens_aug if r["class"] in failed_classes]
    sens_wins = [r for r in paired_wins(sens_groups) if r["class"] in failed_classes]
    tables["sensitivity_generator_fidelity"] = markdown_table(
        sens_gen,
        [
            ("class", "Class"),
            ("method", "Source"),
            ("n", "Seeds"),
            ("mmd", "MMD to validation ↓"),
            ("diversity_ratio", "Diversity (1 = real)"),
            ("euler_best_median", "Physiology (Euler) ↓"),
            ("hf_fraction_median", "Power > 40 Hz"),
        ],
    )
    tables["sensitivity_augmentation"] = markdown_table(
        sens_aug,
        [
            ("class", "Class"),
            ("setup", "Setup"),
            ("amount", "Added"),
            ("n", "Seeds"),
            ("ap", "Test AP"),
            ("gain", "AP gain vs control"),
            ("p_at_r", "Precision @ paper recall"),
            ("ap_strict", "Test AP (no record 202)"),
        ],
    )
    tables["sensitivity_simdcgan_vs_dcgan"] = markdown_table(
        sens_wins,
        [
            ("class", "Class"),
            ("amount", "Added"),
            ("dcgan", "DCGAN AP"),
            ("simdcgan", "SimDCGAN AP"),
            ("wins", "Seeds where SimDCGAN better"),
        ],
    )
    failed_text = "; ".join(
        f"{m} {c} seed {seed} {why}" for (c, m, seed), why in FAILED_RUNS.items()
    )
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
    boot = bootstrap_tables()
    boot_rows = read_csv(TABLES / "bootstrap_setups.csv")
    n_boot_text = f"{int(boot_rows[0]['n_boot']):,} times" if boot_rows else "many times"
    tables["bootstrap_primary"] = markdown_table(
        boot["primary"],
        [
            ("comparison", "Primary comparison (test AP)"),
            ("diff", "Difference [95 % interval]"),
            ("above", "Replicates with difference > 0"),
            ("usable", "Usable replicates"),
        ],
    )
    tables["bootstrap_control"] = markdown_table(
        boot["control"],
        [
            ("class", "Class"),
            ("value", "Control test AP [95 % interval over patients]"),
            ("seed_sd", "Spread over seeds"),
            ("usable", "Usable replicates"),
        ],
    )
    tables["bootstrap_vs_dcgan"] = markdown_table(
        boot["vs_dcgan"],
        [
            ("class", "Class"),
            ("amount", "Added"),
            ("metric", "Measure"),
            ("diff", "SimDCGAN − DCGAN [95 % interval]"),
            ("above", "Replicates > 0"),
            ("usable", "Usable replicates"),
        ],
    )
    tables["bootstrap_vs_control"] = markdown_table(
        boot["vs_control"],
        [
            ("class", "Class"),
            ("setup", "Setup"),
            ("amount", "Added"),
            ("value", "Test AP [95 % interval]"),
            ("diff", "Minus control [95 % interval]"),
            ("above", "Replicates > 0"),
            ("usable", "Usable replicates"),
        ],
    )
    failure_rows = read_csv(TABLES / "failure_study_summary.csv")
    q = lambda r, k, d: f"{float(r[k]):.{d}f} ± {float(r[k + '_sd']):.{d}f}"  # noqa: E731
    tables["failure_study"] = markdown_table(
        [
            {
                "model": r["model"],
                "failed": f"{r['failed']}/{r['runs']}",
                "ci": f"{float(r['ci_low']):.0%}-{float(r['ci_high']):.0%}",
                "mmd": q(r, "mmd", 3),
                "div": q(r, "diversity_ratio", 2),
                "euler": q(r, "euler_best_median", 2),
                "hf": q(r, "hf_fraction_median", 4),
            }
            for r in failure_rows
        ],
        [
            ("model", "Model"),
            ("failed", "Failed runs"),
            ("ci", "Failure rate, 95 % interval"),
            ("mmd", "MMD ↓ (successful runs)"),
            ("div", "Diversity"),
            ("euler", "Physiology (Euler) ↓"),
            ("hf", "Power > 40 Hz"),
        ],
    )
    failure_text = ""
    paired_path = TABLES / "failure_study_paired.json"
    if paired_path.exists():
        paired = json.loads(paired_path.read_text())
        diff = paired["warmup_minus_plain"]
        fisher = ", ".join(
            f"{k.replace('_', ' ')}: p = {v:.2f}" for k, v in paired["fisher_p"].items()
        )
        failure_text = (
            f"Fisher's exact test on the failure counts (descriptive; 10 runs per model can "
            f"only show large differences): {fisher}. Paired by seed, where both succeeded "
            f"({len(paired['seeds'])} seeds), warm-up minus no warm-up: MMD "
            f"{diff['mmd'][0]:+.3f} ± {diff['mmd'][1]:.3f}, Euler score "
            f"{diff['euler_best_median'][0]:+.2f} ± {diff['euler_best_median'][1]:.2f}."
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
    protocol = read_csv(TABLES / "paper_protocol.csv")
    pp = lambda r, part: f"{float(r[part]):.2f} ± {float(r[part + '_sd']):.2f}"  # noqa: E731
    tables["paper_protocol"] = markdown_table(
        [
            {
                "class": r["class"],
                "setup": r["setup"],
                "amount": "—" if r["setup"] == "real" else f"{float(r['amount'])}×",
                "paper": r["paper_p_at_paper_recall"] or "—",
                "whole": pp(r, "p_at_paper_recall_whole"),
                "seen": pp(r, "p_at_paper_recall_seen"),
                "validation": pp(r, "p_at_paper_recall_validation"),
                "unseen": pp(r, "p_at_paper_recall_unseen"),
            }
            for r in protocol
            if r["setup"] in ("real", "dcgan", "simdcgan")
        ],
        [
            ("class", "Class"),
            ("setup", "Setup"),
            ("amount", "Added"),
            ("paper", "Paper (whole DB)"),
            ("whole", "Ours: whole DB"),
            ("seen", "Training patients"),
            ("validation", "Validation patients"),
            ("unseen", "DS2 (unseen)"),
        ],
    )
    repro_path = TABLES / "paper_protocol_reproducibility.json"
    repro_text = ""
    if repro_path.exists():
        repro = json.loads(repro_path.read_text())
        repro_text = (
            f"Reproducibility check: repeating the {repro['models_compared']} fine-tunings "
            f"reproduced the original DS2 AP exactly for {repro['identical']} models; the "
            f"largest difference is {repro['max_abs_diff']:.4f} (rule fixed in advance: at "
            f"most {repro['tolerance']}; "
            f"{'passed' if repro['passed'] else 'exceeded, see below'})."
        )
        if not repro["passed"]:
            repro_text += "\n\n" + INTERPRETATION["reproducibility"]
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
        "`real` row is the reference: 2,000 real *training* beats for MMD, diversity, "
        "physiology and smoothness (a perfect copy of the training distribution would score "
        "these values), and real *validation* beats (other patients) for the distance to the "
        "nearest training beat.",
        "",
        tables["generator_fidelity"],
        "",
        "SimDCGAN vs DCGAN, per seed:",
        "",
        tables["generator_simdcgan_vs_dcgan"],
        "",
        INTERPRETATION["generators"],
        "",
        "## 3b. Training reliability (V, 10 seeds per model)",
        "",
        "Phase 12. Each model was trained on V with seeds 0-9, including SimDCGAN with a "
        "warm-up: the Euler weight rises linearly from 0 over the first 1,000 of 5,000 "
        "iterations (one value, fixed before the runs). A run counts as **failed** when its "
        "best checkpoint has a diversity ratio below 0.3 or more than 10 % of its power above "
        "40 Hz (rule fixed before the runs, decision log §19). Failure rates have exact 95 % "
        "intervals; quality is the mean ± spread over the runs that succeeded.",
        "",
        tables["failure_study"] or "*Not run (summarize_failure_study.py).*",
        "",
        failure_text,
        "",
        INTERPRETATION["reliability"],
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
        INTERPRETATION["augmentation"],
        "",
        "## 4b. Supplementary: without the failed training run (not pre-registered)",
        "",
        f"One of the 18 generator runs failed during training: {failed_text}. The main "
        "tables above keep it, because all three seeds were fixed in advance and a failure "
        "is itself a result. The tables below repeat only the affected class without it, "
        "with the paired comparisons restricted to the remaining seeds. They answer 'how do "
        "the methods compare when training succeeds?' and are not the headline result.",
        "",
        "Generator fidelity:",
        "",
        tables["sensitivity_generator_fidelity"],
        "",
        "Augmentation on DS2 (the failed run's synthetic beats were used only by its own "
        "seed, so that seed is dropped for SimDCGAN):",
        "",
        tables["sensitivity_augmentation"],
        "",
        "SimDCGAN vs DCGAN at the same amount, remaining seeds:",
        "",
        tables["sensitivity_simdcgan_vs_dcgan"],
        "",
        INTERPRETATION["sensitivity"],
        "",
        "## 4c. Uncertainty over test patients (bootstrap)",
        "",
        "The ± values above show how much results change when the classifier is retrained "
        "with another seed. They don't show how much they depend on *which* 22 patients are "
        "in the test set. Here the 22 DS2 records are resampled with replacement "
        f"{n_boot_text}, and every classifier's saved test scores are rescored on each "
        "resample (no retraining). Whole patients are resampled, not single beats, because a "
        "patient's beats are very alike. Metrics are averaged over seeds inside each "
        "resample, and every setup is scored on the same resamples, so differences are "
        "paired. Intervals are 95 % percentile intervals. A resample with no positive beats "
        "of the class can't be scored and is skipped (column *Usable replicates*). The two "
        "primary comparisons were fixed before any interval was computed; every other "
        "interval is descriptive, and with about 70 of them some will exclude 0 by chance.",
        "",
        tables["bootstrap_primary"] or "*Not run (scripts/bootstrap_results.py).*",
        "",
        "Control (real beats only):",
        "",
        tables["bootstrap_control"],
        "",
        "SimDCGAN − DCGAN at the same amount:",
        "",
        tables["bootstrap_vs_dcgan"],
        "",
        "Every setup against the control:",
        "",
        tables["bootstrap_vs_control"],
        "",
        INTERPRETATION["bootstrap"],
        "",
        "## 5. Train on synthetic, test on real",
        "",
        tables["tstr"] or "*Not run.*",
        "",
        INTERPRETATION["tstr"],
        "",
        "## 6. Relation to the SimGANs paper",
        "",
        "The paper's Table 1 trains on DS1 and evaluates on the **whole** MIT-BIH database, "
        "training patients included. To compare directly, the augmentation fine-tunings were "
        "repeated and scored the same way (Phase 11): on the whole database (all 44 records, "
        "100,685 beats), and on each part of it: the training patients (seen), the 3 "
        "validation patients and DS2 (unseen). Precision at the paper's recall, mean ± "
        "spread over 3 seeds:",
        "",
        tables["paper_protocol"] or "*Not run (run_augmentation.py --whole-database).*",
        "",
        repro_text,
        "",
        "Our DS2-only result next to the paper (from section 4, 1×):",
        "",
        tables["paper_comparison"],
        "",
        INTERPRETATION["paper"],
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
        "- The validation set has 3 patients; one (record 118, right bundle branch block, "
        "wide normal beats) dominates pooled validation AP, which is why the pilot rule chose "
        "1 base epoch.",
        "- The classifier sees one beat at a time, without RR-interval timing, so S beats "
        "(recognised clinically by their early timing) are detected at near-chance level; F "
        "training and test beats come from different single patients (records 208 and 213).",
        "- One of the 18 GAN runs (SimDCGAN, V, seed 2) failed at the start of training; it "
        "is kept in the main results and removed only in the labelled supplementary tables. "
        "Over 10 seeds, SimDCGAN without warm-up failed on V in 6 runs (section 3b); the "
        "warm-up removed the failures, but the augmentation experiments were not rerun with "
        "warm-up generators, and the reliability study covers V only.",
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
        "bootstrap_*.png",
        "paper_protocol.png",
        "failure_study.png",
        "api_demo.png",
    ]:
        for path in FIGURES.glob(pattern):
            shutil.copy2(path, REPORT / "figures" / path.name)
    print(f"Wrote {REPORT / 'results.md'} and {len(tables)} tables")


if __name__ == "__main__":
    main()
