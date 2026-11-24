"""Train every generator: classes × models × seeds, skipping runs that already finished.

Run from the repository root:

    .venv/bin/python scripts/train_all.py                       # all 18 runs
    .venv/bin/python scripts/train_all.py --classes S --seeds 0 # a subset

Each run is a separate scripts/train_gan.py process (a clean GPU state per run) named
``<model>_s<seed>``. A run counts as finished when outputs/runs/<name>_<class>/summary.json
exists, so the script can be stopped and restarted without repeating finished work.

Reads:  configs/{dcgan,simdcgan}.yaml
Writes: checkpoints/<model>_s<seed>/<class>/, outputs/runs/<model>_s<seed>_<class>/
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "dcgan": "configs/dcgan.yaml",
    "simdcgan": "configs/simdcgan.yaml",
    "simdcgan_warmup": "configs/simdcgan_warmup.yaml",  # Phase 12 (V only)
    # Phase 15 ablation (V only), each one change from simdcgan_warmup (decision log §22)
    "abl_weight_low": "configs/ablation/weight_low.yaml",
    "abl_weight_high": "configs/ablation/weight_high.yaml",
    "abl_best_of_k": "configs/ablation/best_of_k.yaml",
    "abl_gaussian": "configs/ablation/gaussian.yaml",
    # Phase 16 small-class stabilisation (S, F), decision log §23
    "dcgan_sn": "configs/dcgan_sn.yaml",
    "simdcgan_sn": "configs/simdcgan_sn.yaml",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--classes", nargs="+", default=["S", "V", "F"])
    parser.add_argument("--models", nargs="+", default=["dcgan", "simdcgan"], choices=list(MODELS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = parser.parse_args()

    jobs = [(c, s, m) for c in args.classes for s in args.seeds for m in args.models]
    for number, (beat_type, seed, model) in enumerate(jobs, start=1):
        run_name = f"{model}_s{seed}"
        label = f"[{number}/{len(jobs)}] {run_name} {beat_type}"
        if (ROOT / "outputs" / "runs" / f"{run_name}_{beat_type}" / "summary.json").exists():
            print(f"{label}: already done, skipping", flush=True)
            continue
        began = time.time()
        command = [
            sys.executable, str(ROOT / "scripts" / "train_gan.py"),
            "--config", MODELS[model], "--beat-type", beat_type,
            "--seed", str(seed), "--run-name", run_name,
        ]  # fmt: skip
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"{label}: FAILED\n{result.stderr[-2000:]}", flush=True)
            sys.exit(result.returncode)
        print(f"{label}: done in {(time.time() - began) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
