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

## Train and evaluate the generators

```bash
.venv/bin/python scripts/fit_params.py            # simulator parameters per beat (~4 min, GPU)
.venv/bin/python scripts/train_all.py             # DCGAN and SimDCGAN, S/V/F, 3 seeds (~2 h, GPU)
.venv/bin/python scripts/evaluate_generators.py   # scores + synthetic sets in data/synthetic/
.venv/bin/python scripts/summarize_runs.py        # tables in outputs/tables/
.venv/bin/python scripts/export_models.py         # model bundle for applications
```

Settings are in `configs/fitting.yaml`, `configs/dcgan.yaml` and `configs/simdcgan.yaml`. A single run: `.venv/bin/python scripts/train_gan.py --config configs/simdcgan.yaml --beat-type V`.

## Using SimHeart from Python (for applications)

`simheart.api` gives applications, such as the SimHeart user interface, everything they need without touching the training code. The functions return NumPy arrays and run on the CPU.

**Model bundle.** The API reads a folder made by `scripts/export_models.py` (default `exports/simheart-models/`). It contains the generator weights, the simulator presets and a sample of real validation beats. It isn't in Git (the generators are about 46 MB each), so it's shared separately. Point the API at it once:

```python
from simheart import api

api.set_model_dir("path/to/simheart-models")
api.available_models()  # which generators exist (model, class, seed)
```

**Synthetic beats** from a trained generator (`"simdcgan"` or `"dcgan"`; classes `"S"`, `"V"`, `"F"`):

```python
beats = api.generate("V", n=8, model="simdcgan", random_state=0)
beats.values  # [8, 216] amplitudes, plot against beats.time_ms (-200 ... 397 ms)
beats.synthetic  # True: always show api.SYNTHETIC_NOTICE with synthetic beats
```

**Real beats** for comparison (MIT-BIH validation patients, never used for training):

```python
real = api.real_examples("V", n=8, random_state=0)
```

**Simulator with sliders.** Each wave (P, Q, R, S, T) has an amplitude `a`, a width `b` and a position `theta_deg`:

```python
params = api.preset("V")  # typical values for the class: start the sliders here
ranges = api.slider_ranges("V")  # (low, high) per slider: plausible values for the class
params["R"]["b"] = 0.25  # e.g. widen the QRS complex
beat = api.simulate(params)  # beat.values[0] is the simulated beat
```

**Calibration and what-if.** Fit the simulator to one beat, change one parameter, and simulate again:

```python
fitted = api.calibrate(real.values[0])  # ~3 s on a CPU; fitted["r2"] = fit quality
fitted["params"]["T"]["a"] *= 2  # what if the T wave were twice as tall?
what_if = api.simulate(fitted["params"], fitted["z0"])
```

`scripts/demo_api.py` makes one figure with all of the above. It's a starting point for the interface.

**Responsible use.** Synthetic beats are research artefacts. They are not clinically validated, must not be used for diagnosis, and must never be presented as real patient data. `Beats.synthetic` and `Beats.source` say what every result is.

## Tests

```bash
.venv/bin/python -m pytest
.venv/bin/python -m ruff check .
```

## Data source

Moody GB, Mark RG. *The impact of the MIT-BIH Arrhythmia Database.* IEEE Eng in Med and Biol 20(3):45-50 (2001). Distributed by [PhysioNet](https://physionet.org/content/mitdb/1.0.0/) (Goldberger et al., 2000).
