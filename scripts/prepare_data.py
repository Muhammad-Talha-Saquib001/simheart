"""Download MIT-BIH, cut it into labelled heartbeats and save them with the split lists.

Run from the repository root:

    .venv/bin/python scripts/prepare_data.py

Reads:  configs/data.yaml and the MIT-BIH files in data/raw/mitdb (downloaded if missing)
Writes: data/processed/{beats.npz, splits.json}
        outputs/tables/{class_counts_by_record, class_counts_by_split, excluded_symbols}.csv
        outputs/figures/examples_by_class.png
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from simheart.data import mitbih
from simheart.data.dataset import BEATS_FILE, SPLITS_FILE
from simheart.data.labels import CLASSES
from simheart.data.splits import DS1, DS2, SPLIT_NAMES, build_splits, patient_id, patient_overlap
from simheart.utils import load_config

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_PER_CLASS = 20


def split_of(record: str, splits: dict[str, list[str]]) -> str:
    """Return the split a record belongs to, or "unused" if it is in none."""
    return next((name for name, records in splits.items() if record in records), "unused")


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def plot_examples(
    beats: np.ndarray, labels: np.ndarray, samples_before: int, path: Path, seed: int
) -> None:
    """Overlay a random sample of beats from each class, in millivolts."""
    rng = np.random.default_rng(seed)
    time_ms = (np.arange(beats.shape[1]) - samples_before) / mitbih.SAMPLING_RATE * 1000
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for index, (ax, name) in enumerate(zip(axes.flat, CLASSES, strict=True)):
        members = np.flatnonzero(labels == index)
        chosen = rng.choice(members, size=min(EXAMPLES_PER_CLASS, len(members)), replace=False)
        ax.plot(time_ms, beats[chosen].T, linewidth=0.8, alpha=0.6)
        ax.axvline(0, color="grey", linestyle=":", linewidth=0.8)
        ax.set_title(f"{name}  ({len(members)} beats in DS1, {len(chosen)} shown)")
        ax.set_ylabel("mV")
    for ax in axes[1]:
        ax.set_xlabel("time from R-peak (ms)")
    fig.suptitle("Real MIT-BIH beats (MLII), development records (DS1)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/data.yaml", type=Path)
    args = parser.parse_args()

    config = load_config(ROOT / args.config)
    raw_dir = ROOT / config["raw_dir"]
    processed_dir = ROOT / config["processed_dir"]
    tables_dir = ROOT / config["tables_dir"]
    figures_dir = ROOT / config["figures_dir"]
    before, after = config["samples_before"], config["samples_after"]

    records = sorted(DS1 + DS2)
    print(f"Downloading missing records into {raw_dir} ...")
    mitbih.download(raw_dir, records)

    parts: list[mitbih.ExtractedBeats] = []
    skipped: Counter = Counter()
    for name in records:
        record = mitbih.load_record(raw_dir, name, config["lead"])
        extracted = mitbih.extract_beats(
            record.signal, record.ann_samples, record.ann_symbols, before, after
        )
        parts.append(extracted)
        skipped.update(extracted.skipped)

    record_ids = np.concatenate(
        [np.full(len(p.labels), name) for name, p in zip(records, parts, strict=True)]
    )
    beats = np.concatenate([p.beats for p in parts])
    labels = np.concatenate([p.labels for p in parts])
    processed_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        processed_dir / BEATS_FILE,
        X=beats,
        y=labels,
        record=record_ids,
        patient=np.array([patient_id(r) for r in record_ids]),
        r_peak=np.concatenate([p.r_peaks for p in parts]),
    )

    split_config = config["split"]
    splits = build_splits(split_config["validation_records"], split_config["strict_patients"])
    overlap = sorted(patient_overlap(splits))
    (processed_dir / SPLITS_FILE).write_text(
        json.dumps(
            {
                "records": splits,
                "strict_patients": split_config["strict_patients"],
                "patient_overlap": overlap,
                "lead": config["lead"],
                "samples_before": before,
                "samples_after": after,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    by_record = []
    for name, part in zip(records, parts, strict=True):
        counts = np.bincount(part.labels, minlength=len(CLASSES)).tolist()
        by_record.append([name, patient_id(name), split_of(name, splits), *counts, sum(counts)])
    write_csv(
        tables_dir / "class_counts_by_record.csv",
        ["record", "patient", "split", *CLASSES, "total"],
        by_record,
    )

    by_split = []
    for split_name in [*SPLIT_NAMES, "unused"]:
        rows = [row for row in by_record if row[2] == split_name]
        if rows:
            totals = np.sum([row[3:] for row in rows], axis=0).tolist()
            by_split.append([split_name, len(rows), *totals])
    write_csv(
        tables_dir / "class_counts_by_split.csv",
        ["split", "records", *CLASSES, "total"],
        by_split,
    )
    write_csv(
        tables_dir / "excluded_symbols.csv",
        ["symbol", "reason", "count"],
        [[s, reason, n] for (s, reason), n in sorted(skipped.items())],
    )

    development = np.isin(record_ids, DS1)
    plot_examples(
        beats[development],
        labels[development],
        before,
        figures_dir / "examples_by_class.png",
        config["seed"],
    )

    print(f"Saved {len(labels)} beats to {processed_dir / BEATS_FILE}")
    print(f"{'split':<11}{'records':>8}" + "".join(f"{c:>8}" for c in CLASSES) + f"{'total':>9}")
    for row in by_split:
        print(f"{row[0]:<11}{row[1]:>8}" + "".join(f"{n:>8}" for n in row[2:-1]) + f"{row[-1]:>9}")
    if overlap:
        print(f"Note: patients in more than one split: {overlap} (see split.strict_patients)")


if __name__ == "__main__":
    main()
