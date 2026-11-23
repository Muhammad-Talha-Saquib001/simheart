"""Final evaluation of every trained generator, and the synthetic beat sets.

Run from the repository root, after scripts/train_all.py:

    .venv/bin/python scripts/evaluate_generators.py

For each class and seed, it scores 2,000 beats from each source against the validation
beats, using the same measures as training plus the near-copy rate:

* ``dcgan`` and ``simdcgan``: the run's best checkpoint (checkpoints/<model>_s<seed>/...)
* ``simulator``: ECGSYN beats from parameters drawn with the class's ParamKDE sampler
* ``real``: real training beats (the reference row, computed once per class)

Every synthetic beat is rescaled per beat with ``minmax_to_sim_range``, exactly like real
beats, before it is scored and saved.

Reads:  data/processed/{beats.npz, splits.json, param_samplers.npz}, checkpoints/
Writes: data/synthetic/<method>/<class>_s<seed>.npz (2 × the class's training size)
        outputs/runs/<model>_s<seed>_<class>/final_eval.json
        outputs/tables/generator_eval_per_seed.csv
"""

import argparse
import csv
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from simheart.data.dataset import load_split
from simheart.data.scaling import minmax_to_sim_range, sim_to_unit, unit_to_sim
from simheart.data.synthetic import save_synthetic
from simheart.evaluation import mean_pairwise_distance, mmd_rbf, near_copy_rate, nearest_distances
from simheart.evaluation.physiology import physiology_metrics
from simheart.models.gan import NOISE_DIM
from simheart.simulator import EcgSyn
from simheart.simulator.fitting import load_samplers
from simheart.training.gan import generate, load_generator
from simheart.utils import get_device, set_seed

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
SYNTHETIC = ROOT / "data" / "synthetic"
CHECKPOINTS = ROOT / "checkpoints"
RUNS = ROOT / "outputs" / "runs"
TABLE = ROOT / "outputs" / "tables" / "generator_eval_per_seed.csv"


def load_unit(split: str, beat_type: str) -> np.ndarray:
    beats = load_split(PROCESSED, split, classes=[beat_type])["X"]
    return sim_to_unit(minmax_to_sim_range(beats))


def rescale(beats_sim: np.ndarray) -> np.ndarray:
    """Per-beat min-max into the simulator range, as for real beats."""
    return minmax_to_sim_range(beats_sim.astype(np.float32))


def score(
    beats: torch.Tensor,
    train: torch.Tensor,
    val: torch.Tensor,
    val_diversity: float,
    physiology: Callable[[torch.Tensor], dict[str, float]],
) -> dict[str, float]:
    """All measures for one set of beats ([-1, 1] scale) against the class's real beats."""
    return {
        "mmd": mmd_rbf(beats, val),
        "diversity_ratio": mean_pairwise_distance(beats) / val_diversity,
        "nn_train_median": float(nearest_distances(beats, train).median()),
        "near_copy_rate": near_copy_rate(beats, train),
        **physiology(beats),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--classes", nargs="+", default=["S", "V", "F"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--n-eval", type=int, default=2000)
    args = parser.parse_args()

    device = get_device()
    sim = EcgSyn().to(device)
    samplers = load_samplers(PROCESSED / "param_samplers.npz")
    rows: list[dict] = []

    for beat_type in args.classes:
        train = torch.as_tensor(load_unit("train", beat_type), device=device)
        val = torch.as_tensor(load_unit("validation", beat_type), device=device)
        sampler = samplers[beat_type].to(device)
        physiology = physiology_metrics(sim, sampler, val)
        val_diversity = mean_pairwise_distance(val)
        n_train, n_save = len(train), 2 * len(train)
        n_make = max(args.n_eval, n_save)

        set_seed(0)
        reference = train[torch.randperm(n_train, device=device)[: args.n_eval]]
        real = score(reference, train, val, val_diversity, physiology)
        # Real training beats are trivially "copies" of the training set; for distance
        # measures the honest real reference is the held-out validation beats.
        real["nn_train_median"] = float(nearest_distances(val, train).median())
        real["near_copy_rate"] = near_copy_rate(val, train)
        rows.append({"class": beat_type, "method": "real", "seed": "", **real})
        print(f"{beat_type} real: MMD {real['mmd']:.3f}", flush=True)

        for seed in args.seeds:
            set_seed(1000 + seed)
            sources: dict[str, tuple[np.ndarray, str]] = {}

            generator = torch.Generator(device=device).manual_seed(seed)
            params, z0 = sampler.sample(n_make, generator)
            with torch.no_grad():
                simulated = sim(params, z0).cpu().numpy()
            sources["simulator"] = (rescale(simulated), "ECGSYN with ParamKDE parameters")

            for model in ("dcgan", "simdcgan"):
                checkpoint = CHECKPOINTS / f"{model}_s{seed}" / beat_type / "best.pt"
                net = load_generator(checkpoint, device)
                noise = torch.randn(n_make, NOISE_DIM, device=device)
                made = unit_to_sim(generate(net, noise)).cpu().numpy()
                sources[model] = (rescale(made), str(checkpoint.relative_to(ROOT)))

            for method, (beats_sim, source) in sources.items():
                beats_unit = torch.as_tensor(sim_to_unit(beats_sim[: args.n_eval]), device=device)
                result = score(beats_unit, train, val, val_diversity, physiology)
                rows.append({"class": beat_type, "method": method, "seed": seed, **result})
                save_synthetic(
                    SYNTHETIC, method, beat_type, seed, beats_sim[:n_save], n_train, source
                )
                if method != "simulator":
                    path = RUNS / f"{method}_s{seed}_{beat_type}" / "final_eval.json"
                    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
                print(
                    f"{beat_type} {method} s{seed}: MMD {result['mmd']:.3f}, "
                    f"diversity {result['diversity_ratio']:.2f}, "
                    f"Euler {result['euler_best_median']:.2f}, "
                    f"near-copy {result['near_copy_rate']:.3f}",
                    flush=True,
                )

    TABLE.parent.mkdir(parents=True, exist_ok=True)
    with open(TABLE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {TABLE}")


if __name__ == "__main__":
    main()
