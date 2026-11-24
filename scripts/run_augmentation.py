"""Run the augmentation experiments (and, with --tstr, train-on-synthetic).

Run from the repository root, after scripts/evaluate_generators.py:

    .venv/bin/python scripts/run_augmentation.py
    .venv/bin/python scripts/run_augmentation.py --tstr
    .venv/bin/python scripts/run_augmentation.py --whole-database   # the paper's protocol
    .venv/bin/python scripts/run_augmentation.py --classes V --setups simdcgan_warmup  # Phase 13

For every class and seed: train a base classifier on real training beats, then fine-tune a
copy of it for the same number of epochs with each setup (real-only control, oversampling,
simulator, DCGAN, SimDCGAN) and amount. Every model is scored on validation (to pick a
decision threshold) and on DS2 test beats, with and without record 202. Finished models are
recorded in results.jsonl and skipped on the next run.

With --whole-database (Phase 11), the same augmentation runs are repeated into
outputs/experiments/paper_protocol/ and every model is also scored on the whole database
(train + validation + test records), as the SimGANs paper's Table 1 does, and on each part:
seen (training patients), validation and unseen (DS2) patients.

Reads:  configs/classifier.yaml, data/processed/, data/synthetic/
Writes: outputs/experiments/{results.jsonl, base/, scores/}
        (with --whole-database: outputs/experiments/paper_protocol/, same layout)
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from simheart.classification.datasets import (
    augmented_training_set,
    real_binary,
    strict_mask,
    synthetic_positive_set,
    whole_database,
)
from simheart.classification.metrics import (
    PAPER_RECALL,
    average_precision,
    best_f1_threshold,
    precision_at_recall,
    scores_at_threshold,
)
from simheart.classification.train import ClassifierConfig, predict_scores, train_classifier
from simheart.data.synthetic import ALL_METHODS, METHODS
from simheart.utils import get_device, load_config, set_seed

ROOT = Path(__file__).resolve().parents[1]


def evaluate(state: dict, beat_type: str, sets: dict, device: torch.device) -> tuple[dict, dict]:
    """Scores on validation (threshold) and test (all, and without record 202)."""
    val_scores = predict_scores(state, sets["x_val"], device)
    test_scores = predict_scores(state, sets["x_test"], device)
    threshold = best_f1_threshold(sets["y_val"], val_scores)
    recall = PAPER_RECALL[beat_type]
    result = {"ap_val": average_precision(sets["y_val"], val_scores), "threshold": threshold}
    for suffix, mask in [("", slice(None)), ("_strict", sets["strict"])]:
        y, s = sets["y_test"][mask], test_scores[mask]
        at = scores_at_threshold(y, s, threshold)
        result |= {
            f"ap_test{suffix}": average_precision(y, s),
            f"p_at_paper_recall_test{suffix}": precision_at_recall(y, s, recall),
            f"precision_test{suffix}": at["precision"],
            f"recall_test{suffix}": at["recall"],
            f"f1_test{suffix}": at["f1"],
        }
    scores = {"test": test_scores, "val": val_scores}
    if "x_whole" in sets:  # the paper's protocol: every beat, and each split's part
        whole_scores = predict_scores(state, sets["x_whole"], device)
        split = sets["split"]
        parts = [("_whole", slice(None)), ("_seen", split == "train")]
        parts += [("_validation", split == "validation"), ("_unseen", split == "test")]
        for suffix, mask in parts:
            y, s = sets["y_whole"][mask], whole_scores[mask]
            result |= {
                f"ap{suffix}": average_precision(y, s),
                f"p_at_paper_recall{suffix}": precision_at_recall(y, s, recall),
            }
        scores["whole"] = whole_scores
    return result, scores


def read_results(path: Path) -> list[dict]:
    """Rows of results.jsonl. A cut-off last line (a crash while writing) is moved to
    results.jsonl.partial and dropped, so that model is simply trained again."""
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    rows = []
    for i, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if i != len(lines) - 1:
                raise SystemExit(f"{path}: line {i + 1} is corrupt (not the last line)") from None
            with open(path.with_suffix(".jsonl.partial"), "a", encoding="utf-8") as f:
                f.write(line + "\n")
            good = "".join(kept + "\n" for kept in lines[:-1])  # byte-identical
            tmp = path.with_suffix(".tmp")
            tmp.write_text(good, encoding="utf-8")
            os.replace(tmp, path)
            print(f"Repaired {path}: dropped a cut-off last line (kept in .partial)", flush=True)
            return rows
    if text and not text.endswith("\n") and rows:  # complete JSON but no newline yet
        with open(path, "a", encoding="utf-8") as f:
            f.write("\n")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tstr", action="store_true", help="run train-on-synthetic instead")
    parser.add_argument("--classes", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--settings", type=Path, help="pilot_choice.json (overrides training)")
    parser.add_argument(
        "--amounts", nargs="+", type=float, help="amounts to run instead of the configured ones"
    )
    parser.add_argument(
        "--setups",
        nargs="+",
        choices=["oversample", *ALL_METHODS],
        help="setups (or, with --tstr, methods) to run instead of the configured ones",
    )
    parser.add_argument(
        "--whole-database", action="store_true", help="the paper's protocol (paper_protocol/)"
    )
    args = parser.parse_args()
    if args.tstr and args.whole_database:
        parser.error("--whole-database repeats the augmentation runs only")
    if args.tstr and args.setups and "oversample" in args.setups:
        parser.error("--tstr trains on synthetic beats; oversample is not a synthetic method")

    config = load_config(ROOT / "configs" / "classifier.yaml")
    processed, synthetic = ROOT / config["processed_dir"], ROOT / config["synthetic_dir"]
    out = ROOT / config["out_dir"]
    if args.whole_database:
        out = out / "paper_protocol"
    (out / "base").mkdir(parents=True, exist_ok=True)
    (out / "scores").mkdir(parents=True, exist_ok=True)
    results_path = out / "results.jsonl"
    done = {r["key"] for r in read_results(results_path)}
    training = dict(config["training"])
    if args.settings:
        choice = json.loads(args.settings.read_text(encoding="utf-8"))
        training |= {k: choice[k] for k in ("lr", "base_epochs", "finetune_epochs")}
    settings = ClassifierConfig(**training)
    print(f"Settings: {settings}", flush=True)
    device = get_device()

    def record(row: dict, scores: dict) -> None:
        # Scores first, then the result line, each written so a crash can't leave a
        # half-written file: a model without its line is simply retrained next time.
        score_path = out / "scores" / f"{row['key'].replace('|', '_')}.npz"
        with open(score_path.with_suffix(".tmp"), "wb") as f:
            np.savez(f, **scores)
            f.flush()
            os.fsync(f.fileno())
        os.replace(score_path.with_suffix(".tmp"), score_path)
        with open(results_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
            f.flush()
            os.fsync(f.fileno())
        print(
            f"{row['key']}: AP {row['ap_test']:.3f} (strict {row['ap_test_strict']:.3f}), "
            f"{row['minutes']:.1f} min",
            flush=True,
        )

    for beat_type in args.classes or config["classes"]:
        x_val, y_val, _ = real_binary(processed, "validation", beat_type)
        x_test, y_test, records = real_binary(processed, "test", beat_type)
        sets = {
            "x_val": x_val,
            "y_val": y_val,
            "x_test": x_test,
            "y_test": y_test,
            "strict": strict_mask(records),
        }
        if args.whole_database:
            x_whole, y_whole, _, split = whole_database(processed, beat_type)
            sets |= {"x_whole": x_whole, "y_whole": y_whole, "split": split}
        for seed in args.seeds or config["seeds"]:
            if args.tstr:
                methods = args.setups or METHODS
                jobs = [("tstr", method, config["tstr_amount"]) for method in methods]
            else:
                setups = args.setups or config["setups"]
                jobs = [("augment", "real", 0.0)]
                amounts = args.amounts or config["amounts"]
                jobs += [("augment", s, a) for s in setups for a in amounts]

            base_path = out / "base" / f"{beat_type}_s{seed}.pt"
            if not args.tstr and not base_path.exists():
                set_seed(seed)
                began = time.time()
                x, y = augmented_training_set(processed, synthetic, beat_type, "real", 0, seed)
                state, _ = train_classifier(x, y, settings.base_epochs, settings, device, seed=seed)
                torch.save(state, base_path.with_suffix(".tmp"))
                os.replace(base_path.with_suffix(".tmp"), base_path)
                result, scores = evaluate(state, beat_type, sets, device)
                record(
                    {
                        "key": f"base|{beat_type}|real|0.0|{seed}",
                        "experiment": "base",
                        "class": beat_type,
                        "setup": "real",
                        "amount": 0.0,
                        "seed": seed,
                        "n_train": len(x),
                        "epochs": settings.base_epochs,
                        "lr": settings.lr,
                        "minutes": (time.time() - began) / 60,
                        **result,
                    },
                    scores,
                )
            base_state = None if args.tstr else torch.load(base_path, weights_only=True)

            for experiment, setup, amount in jobs:
                key = f"{experiment}|{beat_type}|{setup}|{amount}|{seed}"
                if key in done:
                    continue
                set_seed(seed)
                began = time.time()
                if experiment == "tstr":
                    x, y = synthetic_positive_set(
                        processed, synthetic, beat_type, setup, amount, seed
                    )
                    state, _ = train_classifier(
                        x, y, settings.base_epochs, settings, device, seed=seed
                    )
                    epochs = settings.base_epochs
                else:
                    x, y = augmented_training_set(
                        processed, synthetic, beat_type, setup, amount, seed
                    )
                    state, _ = train_classifier(
                        x,
                        y,
                        settings.finetune_epochs,
                        settings,
                        device,
                        init_state=base_state,
                        seed=seed,
                    )
                    epochs = settings.finetune_epochs
                result, scores = evaluate(state, beat_type, sets, device)
                record(
                    {
                        "key": key,
                        "experiment": experiment,
                        "class": beat_type,
                        "setup": setup,
                        "amount": amount,
                        "seed": seed,
                        "n_train": len(x),
                        "n_positive": int(y.sum()),
                        "epochs": epochs,
                        "lr": settings.lr,
                        "minutes": (time.time() - began) / 60,
                        **result,
                    },
                    scores,
                )


if __name__ == "__main__":
    main()
