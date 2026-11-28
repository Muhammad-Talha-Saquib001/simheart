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

import math
import os
import threading
from pathlib import Path
from typing import Annotated

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from simheart import api
from simheart.simulator.ecgsyn import WAVES

ROOT = Path(__file__).resolve().parents[2]
MAX_BEATS = 64
NON_FINITE = "These values give a beat with non-finite values; use smaller, finite numbers."
FIT_FAILED = "The fit gave non-finite results; check the beat is not flat and has a normal range."

# JSON allows NaN and Infinity; the simulator and the fitter don't.
FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]

# One fit at a time: each already uses several PyTorch threads, and parallel fits
# compete for the same cores (five at once took ten times longer than five in a row).
_calibrate_lock = threading.Lock()

if "SIMHEART_MODEL_DIR" in os.environ:
    api.set_model_dir(os.environ["SIMHEART_MODEL_DIR"])

app = FastAPI(title="SimHeart console API", version="0.1.0")


@app.exception_handler(RequestValidationError)
def _validation_error(request: Request, error: RequestValidationError) -> JSONResponse:
    """422 with where and what, without echoing the input (it can be large, or NaN)."""
    detail = [{"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]} for e in error.errors()]
    return JSONResponse({"detail": detail}, status_code=422)


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


def _require_finite(beats: api.Beats) -> api.Beats:
    if not np.isfinite(beats.values).all():
        raise HTTPException(400, NON_FINITE)
    return beats


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
    params: dict[str, dict[str, FiniteFloat]]
    z0: FiniteFloat = 0.0


@app.post("/api/simulate")
def simulate(request: SimulateRequest) -> dict:
    if set(request.params) != set(WAVES):
        raise HTTPException(400, f"params needs exactly the waves {WAVES}")
    return _beats_json(_require_finite(_call(api.simulate, request.params, request.z0)))


class CalibrateRequest(BaseModel):
    beat: list[FiniteFloat] = Field(min_length=216, max_length=216)
    steps: int = Field(300, ge=10, le=1500)


@app.post("/api/calibrate")
def calibrate(request: CalibrateRequest) -> dict:
    with _calibrate_lock:
        fitted = _call(api.calibrate, np.asarray(request.beat), request.steps)
    numbers = [
        fitted["z0"],
        fitted["r2"],
        *(v for w in fitted["params"].values() for v in w.values()),
    ]
    if not all(math.isfinite(v) for v in numbers):
        raise HTTPException(400, FIT_FAILED)
    beat = _require_finite(api.simulate(fitted["params"], fitted["z0"]))
    return {**fitted, "fit": _beats_json(beat)}


_dist = ROOT / "frontend" / "dist"
if _dist.is_dir():
    app.mount("/", StaticFiles(directory=_dist, html=True), name="console")
