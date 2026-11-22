"""Show every simheart.api capability in one figure (a template for the user interface).

Run from the repository root, after scripts/export_models.py:

    .venv/bin/python scripts/demo_api.py [--bundle exports/simheart-models] [--beat-type V]

Reads:  the model bundle
Writes: outputs/figures/api_demo.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from simheart import api

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bundle", type=Path, default=ROOT / "exports" / "simheart-models")
    parser.add_argument("--beat-type", default="V", choices=api.BEAT_TYPES)
    args = parser.parse_args()
    api.set_model_dir(args.bundle)
    c = args.beat_type

    real = api.real_examples(c, n=8, random_state=0)
    sim_beats = api.generate(c, n=8, model="simdcgan", random_state=0)
    dc_beats = api.generate(c, n=8, model="dcgan", random_state=0)

    params = api.preset(c)
    simulated = api.simulate(params, beat_type=c)
    wider = {wave: dict(values) for wave, values in params.items()}
    wider["R"]["b"] = api.slider_ranges(c)["R"]["b"][1]  # widest plausible QRS
    simulated_wide = api.simulate(wider, beat_type=c)

    target = real.values[0]
    fitted = api.calibrate(target, steps=600)
    calibrated = api.simulate(fitted["params"], fitted["z0"])
    what_if = {wave: dict(values) for wave, values in fitted["params"].items()}
    what_if["T"]["a"] *= 2
    what_if_beat = api.simulate(what_if, fitted["z0"])

    fig, axes = plt.subplots(2, 3, figsize=(17, 8), sharex=True)
    panels = [
        (axes[0, 0], real, "Real validation beats", "k"),
        (axes[0, 1], dc_beats, "DCGAN (synthetic)", "tab:orange"),
        (axes[0, 2], sim_beats, "SimDCGAN (synthetic)", "tab:red"),
    ]
    for ax, beats, title, color in panels:
        ax.plot(beats.time_ms, beats.values.T, color=color, linewidth=0.8, alpha=0.7)
        ax.set_title(f"{c}: {title}")

    ax = axes[1, 0]
    ax.plot(simulated.time_ms, simulated.values[0], label="class preset")
    ax.plot(simulated.time_ms, simulated_wide.values[0], label="preset, widest QRS")
    ax.set_title("Simulator with slider values")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.plot(real.time_ms, target, "k", label="real beat")
    ax.plot(real.time_ms, calibrated.values[0], "--", label=f"calibrated (R² {fitted['r2']:.3f})")
    ax.set_title("Calibration: simulator fitted to one real beat")
    ax.legend(fontsize=8)

    ax = axes[1, 2]
    ax.plot(real.time_ms, calibrated.values[0], label="calibrated")
    ax.plot(real.time_ms, what_if_beat.values[0], label="what if: T wave ×2")
    ax.set_title("What-if: change one parameter of the calibrated beat")
    ax.legend(fontsize=8)

    for ax in axes[1]:
        ax.set_xlabel("time from R-peak (ms)")
    fig.suptitle(api.SYNTHETIC_NOTICE, fontsize=9)
    fig.tight_layout()
    out = ROOT / "outputs" / "figures" / "api_demo.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
