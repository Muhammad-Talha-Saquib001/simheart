"""Run every step after generator training, in order, with one command.

Run from the repository root, after scripts/train_all.py has finished all 18 runs:

    .venv/bin/python scripts/reproduce_results.py            # all steps (~2-2.5 h on a GPU)
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
    9. build_report_assets.py    outputs/report/ (tables, figures, results.md)

Steps 1-5 are skipped when their output already exists (use --redo to repeat them); steps
6-7 skip finished models themselves; 8-9 always run (they are quick).
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

from simheart.reporting import missing_runs

ROOT = Path(__file__).resolve().parents[1]
PILOT_CHOICE = "outputs/experiments/pilot_choice.json"

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
    ("Report assets", ["build_report_assets.py"], None),
]


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
