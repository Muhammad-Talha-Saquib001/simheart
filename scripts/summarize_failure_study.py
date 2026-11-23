"""Summarise the training-reliability study (Phase 12): failure rates and warm-up effect.

Run from the repository root, after:

    .venv/bin/python scripts/train_all.py --classes V --models dcgan simdcgan simdcgan_warmup \\
        --seeds 0 1 2 3 4 5 6 7 8 9
    .venv/bin/python scripts/summarize_failure_study.py

A run has failed when its best checkpoint is collapsed or noise-like
(simheart.reporting.is_failed_run; rule fixed before the runs, decision log §19). Missing
runs are skipped, so the script also works on a subset.

Reads:  outputs/runs/<model>_s<seed>_<class>/{summary.json, eval_log.csv}
Writes: outputs/tables/failure_study.csv (one row per run),
        outputs/tables/failure_study_summary.csv (one row per model),
        outputs/figures/failure_study.png
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import fisher_exact  # noqa: E402

from simheart.reporting import (  # noqa: E402
    FAILURE_MAX_HF_FRACTION,
    FAILURE_MIN_DIVERSITY,
    clopper_pearson,
    is_failed_run,
    mean_std,
)

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "outputs" / "runs"
TABLES, FIGURES = ROOT / "outputs" / "tables", ROOT / "outputs" / "figures"
MODELS = {"dcgan": "tab:orange", "simdcgan": "tab:red", "simdcgan_warmup": "tab:purple"}
QUALITY = ["mmd", "diversity_ratio", "euler_best_median", "hf_fraction_median"]


def read_run(model: str, seed: int, beat_type: str) -> dict | None:
    folder = RUNS / f"{model}_s{seed}_{beat_type}"
    if not (folder / "summary.json").exists():
        return None
    summary = json.loads((folder / "summary.json").read_text())
    with open(folder / "eval_log.csv", encoding="utf-8") as f:
        evals = list(csv.DictReader(f))
    bad = [
        int(r["iteration"])
        for r in evals
        if is_failed_run({k: float(r[k]) for k in ("diversity_ratio", "hf_fraction_median")})
    ]
    return {
        "model": model,
        "seed": seed,
        "failed": is_failed_run(summary["best"]),
        "first_bad_iteration": bad[0] if bad else "",
        "bad_evaluations": len(bad),
        "evaluations": len(evals),
        "best_iteration": summary["best"]["iteration"],
        **{key: summary["best"][key] for key in QUALITY},
        "mmd_curve": [(int(r["iteration"]), float(r["mmd"])) for r in evals],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--beat-type", default="V")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    args = parser.parse_args()

    runs = [
        run
        for model in MODELS
        for seed in args.seeds
        if (run := read_run(model, seed, args.beat_type)) is not None
    ]
    by_model = {m: [r for r in runs if r["model"] == m] for m in MODELS}
    for model, group in by_model.items():
        failed = [r["seed"] for r in group if r["failed"]]
        print(f"{model}: {len(failed)}/{len(group)} failed (seeds {failed})")

    summary = []
    for model, group in by_model.items():
        if not group:
            continue
        k, n = sum(r["failed"] for r in group), len(group)
        low, high = clopper_pearson(k, n)
        ok = [r for r in group if not r["failed"]]
        row = {
            "model": model,
            "runs": n,
            "failed": k,
            "rate": k / n,
            "ci_low": low,
            "ci_high": high,
        }
        for key in QUALITY:
            row[key], row[f"{key}_sd"] = mean_std(r[key] for r in ok) if ok else (np.nan, np.nan)
        summary.append(row)

    def counts(model: str) -> list[int]:
        group = by_model[model]
        return [sum(r["failed"] for r in group), sum(not r["failed"] for r in group)]

    comparisons = {}
    for a, b in [("simdcgan", "dcgan"), ("simdcgan_warmup", "simdcgan")]:
        if by_model[a] and by_model[b]:
            comparisons[f"{a}_vs_{b}"] = float(fisher_exact([counts(a), counts(b)])[1])
    # Paired by seed: warm-up minus no warm-up, where both runs succeeded.
    plain = {r["seed"]: r for r in by_model["simdcgan"] if not r["failed"]}
    warm = {r["seed"]: r for r in by_model["simdcgan_warmup"] if not r["failed"]}
    shared = sorted(set(plain) & set(warm))
    paired = {
        key: mean_std(warm[s][key] - plain[s][key] for s in shared) if shared else (np.nan, np.nan)
        for key in QUALITY
    }
    for row in summary:
        for name, p in comparisons.items():
            if name.startswith(row["model"] + "_vs_"):
                row["fisher_p_vs_" + name.split("_vs_")[1]] = p

    TABLES.mkdir(parents=True, exist_ok=True)
    per_run = [{k: v for k, v in r.items() if k != "mmd_curve"} for r in runs]
    write_csv(TABLES / "failure_study.csv", per_run)
    write_csv(TABLES / "failure_study_summary.csv", summary)
    (TABLES / "failure_study_paired.json").write_text(
        json.dumps(
            {"seeds": shared, "warmup_minus_plain": paired, "fisher_p": comparisons}, indent=2
        )
    )
    plot(runs, args.beat_type)
    print(f"Fisher p (descriptive): {comparisons}")
    print(f"Wrote {TABLES / 'failure_study_summary.csv'} and {FIGURES / 'failure_study.png'}")


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot(runs: list[dict], beat_type: str) -> None:
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.2))
    left.axvspan(0, FAILURE_MIN_DIVERSITY, color="0.9")
    left.axhspan(FAILURE_MAX_HF_FRACTION, 1.0, color="0.9")
    for model, color in MODELS.items():
        group = [r for r in runs if r["model"] == model]
        left.scatter(
            [r["diversity_ratio"] for r in group],
            [r["hf_fraction_median"] for r in group],
            color=color,
            label=f"{model} ({sum(r['failed'] for r in group)}/{len(group)} failed)",
            alpha=0.8,
        )
        for i, r in enumerate(group):
            its, mmd = zip(*r["mmd_curve"], strict=True)
            right.plot(its, mmd, color=color, alpha=0.6, label=model if i == 0 else None)
    left.set_yscale("log")
    left.set_ylim(3e-4, 1.0)
    left.set_xlabel("diversity ratio at best checkpoint (1 = real)")
    left.set_ylabel("share of power above 40 Hz")
    left.set_title(f"{beat_type}: each run's best checkpoint (grey = failure region)")
    left.legend(fontsize=8)
    right.set_xlabel("iteration")
    right.set_ylabel("validation MMD")
    right.set_title("validation MMD during training, every run")
    right.legend(fontsize=8)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "failure_study.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
