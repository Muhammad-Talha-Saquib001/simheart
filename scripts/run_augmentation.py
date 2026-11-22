"""Run the augmentation experiments (and, with --tstr, train-on-synthetic).

Run from the repository root, after scripts/evaluate_generators.py:

    .venv/bin/python scripts/run_augmentation.py
    .venv/bin/python scripts/run_augmentation.py --tstr

For every class and seed: train a base classifier on real training beats, then fine-tune a
copy of it for the same number of epochs with each setup (real-only control, oversampling,
simulator, DCGAN, SimDCGAN) and amount. Every model is scored on validation (to pick a
decision threshold) and on DS2 test beats, with and without record 202. Finished models are
recorded in results.jsonl and skipped on the next run.

Reads:  configs/classifier.yaml, data/processed/, data/synthetic/
Writes: outputs/experiments/{results.jsonl, base/, scores/}
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from simheart.classification.datasets import (
    augmented_training_set,
    real_binary,
    strict_mask,
    synthetic_positive_set,
)
from simheart.classification.metrics import (
    PAPER_RECALL,
    average_precision,
    best_f1_threshold,
    precision_at_recall,
    scores_at_threshold,
)
from simheart.classification.train import ClassifierConfig, predict_scores, train_classifier
from simheart.data.synthetic import METHODS
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
    return result, {"test": test_scores, "val": val_scores}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tstr", action="store_true", help="run train-on-synthetic instead")
    parser.add_argument("--classes", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    args = parser.parse_args()

    config = load_config(ROOT / "configs" / "classifier.yaml")
    processed, synthetic = ROOT / config["processed_dir"], ROOT / config["synthetic_dir"]
    out = ROOT / config["out_dir"]
    (out / "base").mkdir(parents=True, exist_ok=True)
    (out / "scores").mkdir(parents=True, exist_ok=True)
    results_path = out / "results.jsonl"
    done = set()
    if results_path.exists():
        done = {json.loads(line)["key"] for line in results_path.read_text().splitlines()}
    settings = ClassifierConfig(**config["training"])
    device = get_device()

    def record(row: dict, scores: dict) -> None:
        np.savez(out / "scores" / f"{row['key'].replace('|', '_')}.npz", **scores)
        with open(results_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
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
        for seed in args.seeds or config["seeds"]:
            if args.tstr:
                jobs = [("tstr", method, config["tstr_amount"]) for method in METHODS]
            else:
                jobs = [("augment", "real", 0.0)]
                jobs += [("augment", s, a) for s in config["setups"] for a in config["amounts"]]

            base_path = out / "base" / f"{beat_type}_s{seed}.pt"
            if not args.tstr and not base_path.exists():
                set_seed(seed)
                began = time.time()
                x, y = augmented_training_set(processed, synthetic, beat_type, "real", 0, seed)
                state, _ = train_classifier(x, y, settings.base_epochs, settings, device, seed=seed)
                torch.save(state, base_path)
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
