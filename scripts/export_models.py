"""Export a model bundle for applications (simheart.api) from the trained runs.

Run from the repository root, after scripts/train_all.py and scripts/fit_params.py:

    .venv/bin/python scripts/export_models.py              # seed 0 for every class
    .venv/bin/python scripts/export_models.py --seeds 0 1 2 --default-seed 1

The bundle holds generator weights only (no discriminators, no training state), the
per-class simulator presets and slider ranges, and a sample of real validation beats.

Reads:  checkpoints/<model>_s<seed>/<class>/best.pt, data/processed/
Writes: exports/simheart-models/ (git-ignored; share it with the UI separately)
"""

import argparse
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from simheart.api import BEAT_TYPES, MODELS, write_bundle
from simheart.data.dataset import load_split
from simheart.data.scaling import minmax_to_sim_range
from simheart.simulator.fitting import load_samplers

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
REAL_EXAMPLES_PER_CLASS = 200


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--default-seed", type=int, help="defaults to the first of --seeds")
    parser.add_argument("--out", type=Path, default=ROOT / "exports" / "simheart-models")
    args = parser.parse_args()
    default_seed = args.seeds[0] if args.default_seed is None else args.default_seed

    generators, sources = {}, {}
    for model in MODELS:
        for beat_type in BEAT_TYPES:
            for seed in args.seeds:
                path = ROOT / "checkpoints" / f"{model}_s{seed}" / beat_type / "best.pt"
                checkpoint = torch.load(path, map_location="cpu", weights_only=True)
                generators[(model, beat_type, seed)] = checkpoint["generator"]
                sources[f"{model}_{beat_type}_s{seed}"] = {
                    "checkpoint": str(path.relative_to(ROOT)),
                    "best_iteration": checkpoint["iteration"],
                    "validation_metrics": checkpoint["metrics"],
                }

    fitted = np.load(PROCESSED / "fitted_params.npz")
    fitted_params = {
        c: fitted["params"][(fitted["class"] == c) & fitted["accepted"]] for c in BEAT_TYPES
    }
    space = next(iter(load_samplers(PROCESSED / "param_samplers.npz").values())).space

    rng = np.random.default_rng(0)
    real_beats = {}
    for beat_type in BEAT_TYPES:
        data = load_split(PROCESSED, "validation", classes=[beat_type])
        index = rng.choice(len(data["X"]), min(REAL_EXAMPLES_PER_CLASS, len(data["X"])),
                           replace=False)  # fmt: skip
        real_beats[beat_type] = (minmax_to_sim_range(data["X"][index]), data["record"][index])

    out = write_bundle(
        args.out,
        generators,
        {(m, c): default_seed for m in MODELS for c in BEAT_TYPES},
        fitted_params,
        space,
        real_beats,
        provenance={
            "created": datetime.now(UTC).isoformat(timespec="seconds"),
            "sources": sources,
        },
    )
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    print(f"Wrote {out} ({len(generators)} generators, {size:.0f} MB)")


if __name__ == "__main__":
    main()
