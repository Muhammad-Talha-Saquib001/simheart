"""HTTP server for the SimHeart console (the web user interface in ``frontend/``).

A thin JSON layer over :mod:`simheart.api`; it adds no modelling of its own. Start it from
the repository root:

    .venv/bin/uvicorn simheart.server:app --port 8000

The model bundle is read from ``$SIMHEART_MODEL_DIR`` (default: ``exports/simheart-models``).
Without a bundle the server still starts: ``/api/status`` says so, and the console keeps
working with its in-browser simulator. If ``frontend/dist`` exists (``npm run build``), the
built console is served at ``/``.

Every beat in a response carries ``source`` and ``synthetic``, like :class:`api.Beats`.
"""

import os
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from simheart import api
from simheart.simulator.ecgsyn import WAVES

ROOT = Path(__file__).resolve().parents[2]
MAX_BEATS = 64

if "SIMHEART_MODEL_DIR" in os.environ:
    api.set_model_dir(os.environ["SIMHEART_MODEL_DIR"])

app = FastAPI(title="SimHeart console API", version="0.1.0")


def _beats_json(beats: api.Beats) -> dict:
    info = {k: v for k, v in beats.info.items() if k != "params"}
    return {
        "values": np.round(beats.values.astype(np.float64), 7).tolist(),
        "time_ms": beats.time_ms.tolist(),
        "beat_type": beats.beat_type,
        "source": beats.source,
        "synthetic": beats.synthetic,
        "info": info,
    }


def _bundle_error() -> str | None:
    try:
        api.available_models()
    except FileNotFoundError as error:
        return str(error)
    return None


def _call(fn, *args, **kwargs):
    """Run an api function, turning its errors into HTTP errors the console can show."""
    try:
        return fn(*args, **kwargs)
    except FileNotFoundError as error:
        raise HTTPException(503, f"Model bundle not available: {error}") from error
    except (ValueError, KeyError) as error:
        raise HTTPException(400, str(error)) from error


@app.get("/api/status")
def status() -> dict:
    error = _bundle_error()
    body = {
        "ok": True,
        "bundle": error is None,
        "bundle_error": error,
        "model_dir": str(api.model_dir()),
        "notice": api.SYNTHETIC_NOTICE,
        "beat_types": list(api.BEAT_TYPES),
        "model_names": list(api.MODELS),
        "models": [],
        "attribution": "",
    }
    if error is None:
        body["models"] = [m.__dict__ for m in api.available_models()]
        body["attribution"] = api._manifest().get("data_attribution", "")  # private, same package
        tables = api._simulator_tables()  # private, same package
        body["param_space"] = {
            "b_min": float(tables["b_min"]),
            "theta_r_max": float(tables["theta_r_max"]),
        }
    return body


@app.get("/api/presets")
def presets() -> dict:
    """Slider starting values and ranges for every class in the bundle."""
    return {
        c: {"params": _call(api.preset, c), "ranges": _call(api.slider_ranges, c)}
        for c in api.BEAT_TYPES
    }


@app.get("/api/generate")
def generate(
    beat_type: str = "V",
    n: int = 16,
    model: str = "simdcgan",
    seed: int | None = None,
    random_state: int | None = None,
) -> dict:
    n = max(1, min(n, MAX_BEATS))
    return _beats_json(_call(api.generate, beat_type, n, model, seed, random_state))


@app.get("/api/real")
def real(beat_type: str = "V", n: int = 16, random_state: int | None = None) -> dict:
    n = max(1, min(n, MAX_BEATS))
    return _beats_json(_call(api.real_examples, beat_type, n, random_state))


class SimulateRequest(BaseModel):
    params: dict[str, dict[str, float]]
    z0: float = 0.0


@app.post("/api/simulate")
def simulate(request: SimulateRequest) -> dict:
    if set(request.params) != set(WAVES):
        raise HTTPException(400, f"params needs exactly the waves {WAVES}")
    return _beats_json(_call(api.simulate, request.params, request.z0))


class CalibrateRequest(BaseModel):
    beat: list[float] = Field(min_length=216, max_length=216)
    steps: int = Field(300, ge=10, le=1500)


@app.post("/api/calibrate")
def calibrate(request: CalibrateRequest) -> dict:
    fitted = _call(api.calibrate, np.asarray(request.beat), request.steps)
    beat = api.simulate(fitted["params"], fitted["z0"])
    return {**fitted, "fit": _beats_json(beat)}


_dist = ROOT / "frontend" / "dist"
if _dist.is_dir():
    app.mount("/", StaticFiles(directory=_dist, html=True), name="console")
