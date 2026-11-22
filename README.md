# simheart
Physics-informed GAN for synthetic arrhythmia ECG generation. A PyTorch implementation of SimGANs (ICML 2020) on MIT-BIH.

## Setup

Requires Python 3.11 or newer (developed on 3.14) and, for training, an NVIDIA GPU.
Everything installs into a `.venv` folder inside the repository.

```bash
python3 -m venv .venv
.venv/bin/pip install --no-cache-dir -r requirements.txt
.venv/bin/pip install --no-cache-dir --no-deps -e .
```

`requirements.txt` pins the exact versions used for the reported results.
`--no-cache-dir` stops pip from keeping a copy of the downloads (PyTorch is about 3 GB) in your home folder.

Check that PyTorch sees the GPU:

```bash
.venv/bin/python -c "import torch; print(torch.cuda.is_available())"
```

## Prepare the data

```bash
.venv/bin/python scripts/prepare_data.py
```

This downloads the MIT-BIH Arrhythmia Database (about 100 MB) into `data/raw/`, cuts every N, S, V and F beat into a 216-sample window (200 ms before to 400 ms after the R-peak, lead MLII), and writes:

- `data/processed/beats.npz`: beats in millivolts with their class, record, patient and R-peak position.
- `data/processed/splits.json`: the patient-level train / validation / test record lists.
- `outputs/tables/`: class counts per record and per split, and every annotation that was left out and why.
- `outputs/figures/examples_by_class.png`: example beats of each class.

Settings are in `configs/data.yaml`. `data/` and `outputs/` are not tracked by Git.

## Tests

```bash
.venv/bin/python -m pytest
.venv/bin/python -m ruff check .
```

## Data source

Moody GB, Mark RG. *The impact of the MIT-BIH Arrhythmia Database.* IEEE Eng in Med and Biol 20(3):45-50 (2001). Distributed by [PhysioNet](https://physionet.org/content/mitdb/1.0.0/) (Goldberger et al., 2000).
