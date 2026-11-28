"""The console's HTTP layer (simheart.server) over a small test bundle."""

import json
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import torch

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from simheart import api, server  # noqa: E402
from simheart.models import Generator, init_weights  # noqa: E402
from simheart.simulator import TYPICAL_PARAMS  # noqa: E402
from simheart.simulator.fitting import ParamSpace  # noqa: E402


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    torch.manual_seed(0)
    net = Generator()
    net.apply(init_weights)
    rng = np.random.default_rng(0)
    fitted = {
        c: TYPICAL_PARAMS[c].numpy() * (1 + 0.05 * rng.standard_normal((40, 3, 5)))
        for c in api.BEAT_TYPES
    }
    real = {c: (rng.uniform(-0.0156, 0.0425, (20, 216)), np.array(["118"] * 20)) for c in "SVF"}
    path = api.write_bundle(
        tmp_path_factory.mktemp("bundle"),
        {("simdcgan", "V", 0): net.state_dict()},
        {("simdcgan", "V"): 0},
        fitted,
        ParamSpace(),
        real,
    )
    api.set_model_dir(path)
    yield TestClient(server.app)
    api.set_model_dir(path / "unset")


def test_status_reports_the_bundle_and_notice(client):
    body = client.get("/api/status").json()
    assert body["bundle"] and body["notice"] == api.SYNTHETIC_NOTICE
    assert body["models"] == [
        {"model": "simdcgan", "beat_type": "V", "seed": 0, "is_default": True}
    ]
    assert body["param_space"]["b_min"] == pytest.approx(0.02)


def test_generated_and_real_beats_say_what_they_are(client):
    made = client.get("/api/generate", params={"beat_type": "V", "n": 3, "random_state": 1}).json()
    assert made["synthetic"] and made["source"] == "simdcgan"
    assert np.asarray(made["values"]).shape == (3, 216) and len(made["time_ms"]) == 216
    real = client.get("/api/real", params={"beat_type": "V", "n": 2}).json()
    assert not real["synthetic"] and real["source"] == "real"
    assert real["info"]["records"] == ["118"] * 2


def test_bad_requests_are_400(client):
    assert client.get("/api/generate", params={"beat_type": "X"}).status_code == 400
    assert client.get("/api/generate", params={"beat_type": "S"}).status_code == 400
    missing_seed = client.get("/api/generate", params={"beat_type": "V", "seed": 9})
    assert missing_seed.status_code == 400
    assert "available seeds: 0" in missing_seed.json()["detail"]
    assert str(api.model_dir()) not in missing_seed.json()["detail"]


def _json(body: dict) -> dict:
    """Send NaN and Infinity as JSON allows them (httpx's json= refuses to)."""
    return {"content": json.dumps(body), "headers": {"content-type": "application/json"}}


def test_non_finite_numbers_are_rejected(client):
    params = client.get("/api/presets").json()["V"]["params"]
    nan_amplitude = {w: dict(p) for w, p in params.items()}
    nan_amplitude["R"]["a"] = float("nan")
    assert client.post("/api/simulate", **_json({"params": nan_amplitude})).status_code == 422
    assert (
        client.post("/api/simulate", **_json({"params": params, "z0": math.inf})).status_code == 422
    )
    beat = client.post("/api/simulate", json={"params": params}).json()["values"][0]
    nan_beat = {"beat": [math.nan, *beat[1:]], "steps": 10}
    response = client.post("/api/calibrate", **_json(nan_beat))
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "beat", 0]
    assert "input" not in response.json()["detail"][0]  # never echoed back
    # Finite inputs that overflow inside the model are a bad request, not a server error.
    huge_width = {w: dict(p) for w, p in params.items()}
    huge_width["R"]["b"] = 1e308  # beyond 32-bit floats
    response = client.post("/api/simulate", json={"params": huge_width})
    assert response.status_code == 400 and "32-bit" in response.json()["detail"]
    huge_height = {w: dict(p) for w, p in params.items()}
    huge_height["R"]["a"] = 3e38  # fits, but the beat overflows
    response = client.post("/api/simulate", json={"params": huge_height})
    assert response.status_code == 400 and "non-finite" in response.json()["detail"]
    huge_beat = {"beat": [1e300] * 108 + [-1e300] * 108, "steps": 10}
    assert client.post("/api/calibrate", json=huge_beat).status_code == 400
    flat = client.post("/api/calibrate", json={"beat": [0.0] * 216, "steps": 10})
    assert flat.status_code == 400 and "not flat" in flat.json()["detail"]


def test_calibrations_run_one_at_a_time(client, monkeypatch):
    params = client.get("/api/presets").json()["V"]["params"]
    beat = client.post("/api/simulate", json={"params": params}).json()["values"][0]
    running, most = 0, 0
    count = threading.Lock()

    def slow_fit(beat, steps):
        nonlocal running, most
        with count:
            running += 1
            most = max(most, running)
        time.sleep(0.2)
        with count:
            running -= 1
        return {"params": params, "z0": 0.0, "r2": 1.0}

    monkeypatch.setattr(api, "calibrate", slow_fit)
    with ThreadPoolExecutor(3) as pool:
        codes = list(
            pool.map(
                lambda _: client.post("/api/calibrate", json={"beat": beat}).status_code, range(3)
            )
        )
    assert codes == [200] * 3 and most == 1


def test_presets_simulate_and_calibrate_round_trip(client):
    presets = client.get("/api/presets").json()
    assert set(presets) == {"S", "V", "F"}
    params = presets["V"]["params"]
    beat = client.post("/api/simulate", json={"params": params}).json()["values"][0]
    fitted = client.post("/api/calibrate", json={"beat": beat, "steps": 150}).json()
    assert fitted["r2"] > 0.9 and len(fitted["fit"]["values"][0]) == 216


def test_status_without_a_bundle_still_answers(client, tmp_path):
    previous = api.model_dir()
    api.set_model_dir(tmp_path)
    try:
        body = client.get("/api/status").json()
        assert body["ok"] and not body["bundle"] and "not found" in body["bundle_error"]
        assert client.get("/api/real").status_code == 503
        assert client.get("/api/generate", params={"seed": 0}).status_code == 503
    finally:
        api.set_model_dir(previous)
