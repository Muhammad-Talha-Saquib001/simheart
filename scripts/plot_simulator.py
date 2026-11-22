"""Plot each class's typical simulated beat next to the real beats of that class.

Run from the repository root, after scripts/prepare_data.py:

    .venv/bin/python scripts/plot_simulator.py

Reads:  data/processed/{beats.npz, splits.json}
Writes: outputs/figures/simulator_typical_vs_real.png
Prints: the raw amplitude range of each simulated beat, to compare with the range that
        data.scaling maps real beats into.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from simheart.data.dataset import load_split
from simheart.data.labels import CLASSES
from simheart.data.scaling import SIM_MAX, SIM_MIN, minmax_to_sim_range
from simheart.simulator import MCSHARRY_PARAMS, TYPICAL_PARAMS, EcgSyn

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
FIGURE = ROOT / "outputs" / "figures" / "simulator_typical_vs_real.png"
SAMPLES_BEFORE = 72
SAMPLING_RATE = 360


def main() -> None:
    sim = EcgSyn()
    with torch.no_grad():
        typical = {c: sim(TYPICAL_PARAMS[c][None])[0].numpy() for c in CLASSES}
        mcsharry = sim(MCSHARRY_PARAMS[None])[0].numpy()

    print(f"Real beats are scaled into [{SIM_MIN:.5f}, {SIM_MAX:.5f}]")
    for name, beat in [("McSharry", mcsharry), *typical.items()]:
        print(f"  simulated {name:<9} raw range [{beat.min():.5f}, {beat.max():.5f}]")

    train = load_split(PROCESSED, "train")
    real = minmax_to_sim_range(train["X"])
    time_ms = (np.arange(real.shape[1]) - SAMPLES_BEFORE) / SAMPLING_RATE * 1000

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
    for index, (ax, name) in enumerate(zip(axes.flat, CLASSES, strict=True)):
        members = real[train["y"] == index]
        low, mid, high = np.percentile(members, [10, 50, 90], axis=0)
        ax.fill_between(time_ms, low, high, color="tab:blue", alpha=0.2, label="real 10-90%")
        ax.plot(time_ms, mid, color="tab:blue", label=f"real median (n={len(members)})")
        ax.plot(time_ms, typical[name], color="tab:red", label="simulator, typical params")
        if name == "N":
            ax.plot(time_ms, mcsharry, "k--", linewidth=1, label="simulator, McSharry 2003")
        ax.set_title(name)
        ax.legend(fontsize=7, loc="upper right")
    for ax in axes[1]:
        ax.set_xlabel("time from R-peak (ms)")
    for ax in axes[:, 0]:
        ax.set_ylabel("scaled amplitude")
    fig.suptitle("ECGSYN typical beats (raw output) vs real training beats (scaled per beat)")
    fig.tight_layout()
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, dpi=120)
    plt.close(fig)
    print(f"Saved {FIGURE}")


if __name__ == "__main__":
    main()
