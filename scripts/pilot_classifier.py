"""Tune the real-only classifier on validation beats before the main experiments.

Run from the repository root:

    .venv/bin/python scripts/pilot_classifier.py

Trains the base classifier (real training beats only) for up to ``pilot.max_epochs`` at
each learning rate and records the validation average precision after every epoch. The
test set is not touched. Copy the chosen epochs and learning rate into
configs/classifier.yaml (``training``) before running scripts/run_augmentation.py.

Reads:  configs/classifier.yaml, data/processed/
Writes: outputs/experiments/pilot.csv, outputs/experiments/pilot.png
"""

import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt

from simheart.classification.datasets import real_binary
from simheart.classification.metrics import average_precision
from simheart.classification.train import ClassifierConfig, predict_scores, train_classifier
from simheart.utils import get_device, load_config, set_seed

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    config = load_config(ROOT / "configs" / "classifier.yaml")
    pilot = config["pilot"]
    processed = ROOT / config["processed_dir"]
    out = ROOT / config["out_dir"]
    out.mkdir(parents=True, exist_ok=True)
    device = get_device()
    rows = []

    for beat_type in pilot["classes"]:
        x_train, y_train, _ = real_binary(processed, "train", beat_type)
        x_val, y_val, _ = real_binary(processed, "validation", beat_type)
        for lr in pilot["lrs"]:
            set_seed(pilot["seed"])
            settings = ClassifierConfig(lr=lr, batch_size=config["training"]["batch_size"])
            began = time.time()

            def score(
                epoch, model, beat_type=beat_type, lr=lr, began=began, x_val=x_val, y_val=y_val
            ):
                ap = average_precision(y_val, predict_scores(model, x_val, device))
                rows.append(
                    {
                        "class": beat_type,
                        "lr": lr,
                        "epoch": epoch,
                        "val_ap": ap,
                        "seconds": round(time.time() - began, 1),
                    }
                )

            train_classifier(
                x_train,
                y_train,
                pilot["max_epochs"],
                settings,
                device,
                seed=pilot["seed"],
                on_epoch=score,
            )
            best = max(
                (r for r in rows if r["class"] == beat_type and r["lr"] == lr),
                key=lambda r: r["val_ap"],
            )
            print(
                f"{beat_type} lr {lr}: best validation AP {best['val_ap']:.3f} at epoch "
                f"{best['epoch']} ({rows[-1]['seconds']:.0f} s for {pilot['max_epochs']} "
                "epochs)",
                flush=True,
            )

    with open(out / "pilot.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(1, len(pilot["classes"]), figsize=(6 * len(pilot["classes"]), 4))
    for ax, beat_type in zip(list(axes), pilot["classes"], strict=True):
        for lr in pilot["lrs"]:
            sel = [r for r in rows if r["class"] == beat_type and r["lr"] == lr]
            ax.plot([r["epoch"] for r in sel], [r["val_ap"] for r in sel], "o-", label=f"lr {lr}")
        ax.set_title(f"{beat_type}: real-only classifier, validation AP")
        ax.set_xlabel("epoch")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out / "pilot.png", dpi=110)
    print(f"Saved {out / 'pilot.csv'}")


if __name__ == "__main__":
    main()
