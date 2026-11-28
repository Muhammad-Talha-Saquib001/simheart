"""The console's HTTP layer (simheart.server) over a small test bundle."""

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


def test_bad_requests_are_400_and_missing_generators_503(client):
    assert client.get("/api/generate", params={"beat_type": "X"}).status_code == 400
    assert client.get("/api/generate", params={"beat_type": "S"}).status_code == 400
    assert client.get("/api/generate", params={"beat_type": "V", "seed": 9}).status_code == 503


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
    finally:
        api.set_model_dir(previous)
