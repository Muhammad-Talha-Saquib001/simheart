"""Run every step after generator training, in order, with one command.

Run from the repository root, after scripts/train_all.py has finished all 18 runs:

    .venv/bin/python scripts/reproduce_results.py            # all steps (~10-11 h on a GPU)
    .venv/bin/python scripts/reproduce_results.py --check    # only check training is complete

Steps (each a separate process; the script stops at the first failure):

    1. evaluate_generators.py    final generator scores + synthetic beat sets
    2. summarize_runs.py         generator comparison tables and figures
    3. export_models.py          model bundle for the UI (default seed chosen on validation)
    4. demo_api.py               API demo figure
    5. pilot_classifier.py       classifier settings, chosen on validation by a fixed rule
    6. run_augmentation.py       augmentation experiments (resumable)
    7. run_augmentation.py --tstr   train-on-synthetic (resumable)
    8. summarize_augmentation.py tables and figures
    9. bootstrap_results.py      confidence intervals over test patients (CPU, no retraining)
   10. run_augmentation.py --whole-database   the paper's protocol (resumable)
   11. summarize_paper_protocol.py            its tables, figure and reproducibility check
   12. train_all.py (V, 3 models x 10 seeds)  training reliability study (~2.6 h, resumable)
   13. summarize_failure_study.py             failure rates and the warm-up's effect
   14. evaluate_generators.py (V, warm-up)    warm-up generator scores + synthetic sets
   15. run_augmentation.py (V, warm-up)       supplementary augmentation (resumable)
   16. run_augmentation.py --tstr (V, warm-up)  supplementary train-on-synthetic
   17. summarize_warmup_augmentation.py       its tables, intervals and figure (CPU)
   18-21. V over 10 generator seeds: evaluate seeds 3-9, augmentation, train-on-synthetic,
          summary (Phase 14)
   22-26. Euler-loss ablation on V: 40 GAN runs (~4.3 h), evaluation, augmentation at 1x,
          train-on-synthetic, summary (Phase 15)
   27-29. Small-class stabilisation (S, F): 12 GAN runs (~1.3 h), evaluation, summary
          (Phase 16)
   30. build_report_assets.py    outputs/report/ (tables, figures, results.md)

Steps whose output is named in STEPS are skipped when it exists (use --redo to repeat them);
training and classifier steps skip finished models themselves; the rest always run. For a
long unattended run of the Phase 14-16 steps, use scripts/run_queue.py (step markers, a
memory watchdog and a progress log).
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

from simheart.reporting import missing_runs

ROOT = Path(__file__).resolve().parents[1]
PILOT_CHOICE = "outputs/experiments/pilot_choice.json"
SEEDS_3_9 = [str(s) for s in range(3, 10)]
SEEDS_0_9 = [str(s) for s in range(10)]
ABLATION = ["abl_weight_low", "abl_weight_high", "abl_best_of_k", "abl_gaussian"]

# (description, script arguments, output that marks the step as done or None = always run)
STEPS: list[tuple[str, list[str], str | None]] = [
    (
        "Evaluate generators",
        ["evaluate_generators.py"],
        "outputs/tables/generator_eval_per_seed.csv",
    ),
    ("Summarise generators", ["summarize_runs.py"], "outputs/tables/generator_comparison.csv"),
    (
        "Export model bundle",
        ["export_models.py", "--seeds", "0", "1", "2", "--default-from-eval", "--defaults-only"],
        "exports/simheart-models/manifest.json",
    ),
    ("API demo", ["demo_api.py"], "outputs/figures/api_demo.png"),
    ("Classifier pilot", ["pilot_classifier.py"], PILOT_CHOICE),
    ("Augmentation experiments", ["run_augmentation.py", "--settings", PILOT_CHOICE], None),
    ("Train on synthetic", ["run_augmentation.py", "--tstr", "--settings", PILOT_CHOICE], None),
    ("Summarise experiments", ["summarize_augmentation.py"], None),
    ("Bootstrap intervals", ["bootstrap_results.py"], "outputs/tables/bootstrap_setups.csv"),
    (
        "Paper protocol",
        ["run_augmentation.py", "--whole-database", "--settings", PILOT_CHOICE],
        None,
    ),
    ("Summarise paper protocol", ["summarize_paper_protocol.py"], None),
    (
        "Reliability study training",
        [
            "train_all.py",
            "--classes",
            "V",
            "--models",
            "dcgan",
            "simdcgan",
            "simdcgan_warmup",
            "--seeds",
            *[str(s) for s in range(10)],
        ],
        None,
    ),  # fmt: skip
    ("Summarise reliability study", ["summarize_failure_study.py"], None),
    (
        "Evaluate warm-up generators",
        [
            "evaluate_generators.py",
            "--classes",
            "V",
            "--models",
            "simdcgan_warmup",
            "--table",
            "outputs/tables/generator_eval_warmup.csv",
        ],
        "outputs/tables/generator_eval_warmup.csv",
    ),  # fmt: skip
    (
        "Warm-up augmentation",
        [
            "run_augmentation.py",
            "--classes",
            "V",
            "--setups",
            "simdcgan_warmup",
            "--settings",
            PILOT_CHOICE,
        ],
        None,
    ),  # fmt: skip
    (
        "Warm-up train on synthetic",
        [
            "run_augmentation.py",
            "--tstr",
            "--classes",
            "V",
            "--setups",
            "simdcgan_warmup",
            "--settings",
            PILOT_CHOICE,
        ],
        None,
    ),  # fmt: skip
    (
        "Summarise warm-up augmentation",
        ["summarize_warmup_augmentation.py"],
        "outputs/tables/warmup_comparisons.csv",
    ),
    # Phase 14: the main V comparison over 10 generator seeds (decision log §21).
    (
        "V ten seeds: evaluate generators",
        [
            "evaluate_generators.py",
            "--classes",
            "V",
            "--seeds",
            *SEEDS_3_9,
            "--models",
            "simulator",
            "dcgan",
            "simdcgan",
            "simdcgan_warmup",
            "--table",
            "outputs/tables/generator_eval_v_seeds3_9.csv",
        ],
        "outputs/tables/generator_eval_v_seeds3_9.csv",
    ),  # fmt: skip
    (
        "V ten seeds: augmentation",
        [
            "run_augmentation.py",
            "--classes",
            "V",
            "--seeds",
            *SEEDS_3_9,
            "--setups",
            "oversample",
            "simulator",
            "dcgan",
            "simdcgan",
            "simdcgan_warmup",
            "--settings",
            PILOT_CHOICE,
        ],
        None,
    ),  # fmt: skip
    (
        "V ten seeds: train on synthetic",
        [
            "run_augmentation.py",
            "--tstr",
            "--classes",
            "V",
            "--seeds",
            *SEEDS_3_9,
            "--setups",
            "simulator",
            "dcgan",
            "simdcgan",
            "simdcgan_warmup",
            "--settings",
            PILOT_CHOICE,
        ],
        None,
    ),  # fmt: skip
    (
        "V ten seeds: summary",
        ["summarize_v_ten_seeds.py"],
        "outputs/tables/v_ten_seeds_comparisons.csv",
    ),  # fmt: skip
    # Phase 15: Euler-loss ablation on V (decision log §22).
    (
        "Ablation: training",
        ["train_all.py", "--classes", "V", "--models", *ABLATION, "--seeds", *SEEDS_0_9],
        None,
    ),
    (
        "Ablation: evaluate generators",
        [
            "evaluate_generators.py",
            "--classes",
            "V",
            "--seeds",
            *SEEDS_0_9,
            "--models",
            *ABLATION,
            "--table",
            "outputs/tables/generator_eval_ablation.csv",
        ],
        "outputs/tables/generator_eval_ablation.csv",
    ),  # fmt: skip
    (
        "Ablation: augmentation",
        [
            "run_augmentation.py",
            "--classes",
            "V",
            "--seeds",
            *SEEDS_0_9,
            "--setups",
            *ABLATION,
            "--amounts",
            "1.0",
            "--settings",
            PILOT_CHOICE,
        ],
        None,
    ),  # fmt: skip
    (
        "Ablation: train on synthetic",
        [
            "run_augmentation.py",
            "--tstr",
            "--classes",
            "V",
            "--seeds",
            *SEEDS_0_9,
            "--setups",
            *ABLATION,
            "--settings",
            PILOT_CHOICE,
        ],
        None,
    ),  # fmt: skip
    ("Ablation: summary", ["summarize_ablation.py"], "outputs/tables/ablation_comparisons.csv"),
    # Phase 16: small-class GAN stabilisation (decision log §23).
    (
        "Stabilisation: training",
        [
            "train_all.py",
            "--classes",
            "S",
            "F",
            "--models",
            "dcgan_sn",
            "simdcgan_sn",
            "--seeds",
            "0",
            "1",
            "2",
        ],
        None,
    ),  # fmt: skip
    (
        "Stabilisation: evaluate generators",
        [
            "evaluate_generators.py",
            "--classes",
            "S",
            "F",
            "--seeds",
            "0",
            "1",
            "2",
            "--models",
            "dcgan_sn",
            "simdcgan_sn",
            "--table",
            "outputs/tables/generator_eval_sn.csv",
        ],
        "outputs/tables/generator_eval_sn.csv",
    ),  # fmt: skip
    ("Stabilisation: summary", ["summarize_small_class_fix.py"], None),
    ("Report assets", ["build_report_assets.py"], None),
]
# The first step of the Phase 14-16 batch (scripts/run_queue.py runs from here).
FIRST_BATCH_STEP = "V ten seeds: evaluate generators"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="only check training is complete")
    parser.add_argument("--redo", action="store_true", help="repeat steps whose output exists")
    args = parser.parse_args()

    missing = missing_runs(ROOT / "outputs" / "runs")
    if missing:
        print(f"Training is not complete: {len(missing)} of 18 runs unfinished:")
        print("  " + ", ".join(missing))
        print("Finish them with: .venv/bin/python scripts/train_all.py")
        sys.exit(1)
    print("All 18 training runs are complete.")
    if args.check:
        return

    for number, (description, command, output) in enumerate(STEPS, start=1):
        label = f"[{number}/{len(STEPS)}] {description}"
        if output and (ROOT / output).exists() and not args.redo:
            print(f"{label}: already done ({output}), skipping", flush=True)
            continue
        print(f"{label}: running {' '.join(command)}", flush=True)
        began = time.time()
        script, *rest = command
        result = subprocess.run([sys.executable, str(ROOT / "scripts" / script), *rest], cwd=ROOT)
        if result.returncode != 0:
            print(f"{label}: FAILED (exit code {result.returncode}); fix and rerun to resume")
            sys.exit(result.returncode)
        print(f"{label}: done in {(time.time() - began) / 60:.1f} min", flush=True)
    print("All steps done. Results: outputs/report/")


if __name__ == "__main__":
    main()
