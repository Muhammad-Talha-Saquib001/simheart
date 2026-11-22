"""Reading MIT-BIH records and cutting them into fixed-length heartbeats."""

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import wfdb

from simheart.data.labels import CLASS_TO_INDEX, aami_class, is_beat_symbol

SAMPLING_RATE = 360  # Hz, for every MIT-BIH record

# Reasons a beat annotation does not become a training example.
NON_BEAT = "non_beat"  # the symbol marks an event, not a heartbeat
EXCLUDED_CLASS = "excluded_class"  # a heartbeat outside N/S/V/F (paced or unclassifiable)
AT_EDGE = "at_edge"  # the window would run past the start or end of the recording


@dataclass
class Record:
    """One lead of a MIT-BIH record and its beat annotations."""

    name: str
    signal: np.ndarray  # shape [n_samples], millivolts
    ann_samples: np.ndarray  # annotation positions, in samples
    ann_symbols: list[str]


@dataclass
class ExtractedBeats:
    """Beats cut from one record, plus counts of annotations that were left out."""

    beats: np.ndarray  # shape [n_beats, before + after], millivolts
    labels: np.ndarray  # class indices into labels.CLASSES
    r_peaks: np.ndarray  # R-peak position of each beat in the recording, in samples
    skipped: Counter = field(default_factory=Counter)  # (symbol, reason) -> count


RECORD_FILES = (".hea", ".dat", ".atr")  # header, signal and reference annotations


def download(raw_dir: Path, records: list[str]) -> None:
    """Download the given records from PhysioNet into raw_dir, unless already complete.

    A record counts as present only if all of its files exist, so an interrupted
    download is completed on the next run.
    """
    missing = [
        r for r in records if not all((raw_dir / f"{r}{ext}").exists() for ext in RECORD_FILES)
    ]
    if missing:
        raw_dir.mkdir(parents=True, exist_ok=True)
        wfdb.dl_database("mitdb", dl_dir=str(raw_dir), records=missing)


def load_record(raw_dir: Path, name: str, lead: str = "MLII") -> Record:
    """Load one lead of a record by channel name, together with its annotations."""
    path = str(raw_dir / name)
    record = wfdb.rdrecord(path)
    if record.fs != SAMPLING_RATE:
        raise ValueError(f"Record {name}: expected {SAMPLING_RATE} Hz, got {record.fs}")
    if lead not in record.sig_name:
        raise ValueError(f"Record {name}: lead {lead!r} not found in {record.sig_name}")
    signal = record.p_signal[:, record.sig_name.index(lead)].astype(np.float32)

    annotation = wfdb.rdann(path, "atr")
    return Record(name, signal, np.asarray(annotation.sample), list(annotation.symbol))


def extract_beats(
    signal: np.ndarray,
    ann_samples: np.ndarray,
    ann_symbols: list[str],
    samples_before: int = 72,
    samples_after: int = 144,
) -> ExtractedBeats:
    """Cut a window around every N/S/V/F beat annotation.

    Each window is ``signal[r - samples_before : r + samples_after]``, so the R-peak
    sits at index ``samples_before``. Every annotation that is not kept is counted in
    ``skipped`` by symbol and reason.
    """
    length = samples_before + samples_after
    beats, labels, r_peaks = [], [], []
    skipped: Counter = Counter()

    for r_peak, symbol in zip(ann_samples, ann_symbols, strict=True):
        if not is_beat_symbol(symbol):
            skipped[(symbol, NON_BEAT)] += 1
            continue
        beat_class = aami_class(symbol)
        if beat_class is None:
            skipped[(symbol, EXCLUDED_CLASS)] += 1
            continue
        start, end = int(r_peak) - samples_before, int(r_peak) + samples_after
        if start < 0 or end > len(signal):
            skipped[(symbol, AT_EDGE)] += 1
            continue
        beats.append(signal[start:end])
        labels.append(CLASS_TO_INDEX[beat_class])
        r_peaks.append(int(r_peak))

    return ExtractedBeats(
        beats=np.asarray(beats, dtype=np.float32).reshape(-1, length),
        labels=np.asarray(labels, dtype=np.int64),
        r_peaks=np.asarray(r_peaks, dtype=np.int64),
        skipped=skipped,
    )
