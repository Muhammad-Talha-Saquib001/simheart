import numpy as np
import pytest
import torch

from simheart import api
from simheart.models import Generator, init_weights
from simheart.simulator import TYPICAL_PARAMS
from simheart.simulator.fitting import ParamSpace


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    """One small bundle for the module: untrained V generators (each is ~46 MB, so only two),
    fitted parameters near the typical beats, and random "real" beats."""
    torch.manual_seed(0)
    generators = {}
    for model in api.MODELS:
        net = Generator()
        net.apply(init_weights)
        generators[(model, "V", 0)] = net.state_dict()
    rng = np.random.default_rng(0)
    fitted = {
        c: TYPICAL_PARAMS[c].numpy() * (1 + 0.05 * rng.standard_normal((50, 3, 5)))
        for c in api.BEAT_TYPES
    }
    real = {
        c: (rng.uniform(-0.0156, 0.0425, (30, 216)), np.array(["118"] * 30)) for c in api.BEAT_TYPES
    }
    path = api.write_bundle(
        tmp_path_factory.mktemp("bundle"),
        generators,
        {(m, "V"): 0 for m in api.MODELS},
        fitted,
        ParamSpace(),
        real,
    )
    api.set_model_dir(path)
    yield path
    api.set_model_dir(path / "unset")


@pytest.fixture(autouse=True)
def use_bundle(request):
    """Re-select the shared bundle before each test (one test points the API elsewhere)."""
    if "bundle" in request.fixturenames:
        api.set_model_dir(request.getfixturevalue("bundle"))


def test_available_models_lists_every_generator_with_its_default(bundle):
    models = api.available_models()
    assert {(m.model, m.beat_type) for m in models} == {("simdcgan", "V"), ("dcgan", "V")}
    assert all(m.is_default and m.seed == 0 for m in models)


def test_generate_is_labelled_reproducible_and_rescaled_like_real_beats(bundle):
    first = api.generate("V", n=5, random_state=7)
    again = api.generate("V", n=5, random_state=7)
    assert first.values.shape == (5, 216) and first.time_ms.shape == (216,)
    assert first.synthetic and first.source == "simdcgan" and first.info["seed"] == 0
    np.testing.assert_array_equal(first.values, again.values)
    np.testing.assert_allclose(first.values.min(axis=1), -0.01563, atol=1e-6)
    np.testing.assert_allclose(first.values.max(axis=1), 0.042557, atol=1e-6)


def test_preset_lies_within_slider_ranges_and_simulates_a_beat_at_the_r_peak(bundle):
    params = api.preset("S")
    ranges = api.slider_ranges("S")
    for wave, fields in params.items():
        for name, value in fields.items():
            low, high = ranges[wave][name]
            assert low <= value <= high
    beat = api.simulate(params)
    assert beat.values.shape == (1, 216) and beat.synthetic
    assert abs(int(np.argmax(beat.values[0])) - api.R_PEAK_INDEX) <= 2


def test_changing_the_t_wave_only_changes_the_end_of_the_beat(bundle):
    params = api.preset("V")
    base = api.simulate(params).values[0]
    params["T"]["a"] *= 3
    changed = api.simulate(params).values[0]
    difference = np.abs(changed - base)
    assert difference[:100].max() < 0.1 * difference.max()


def test_parameter_dictionaries_round_trip():
    params = TYPICAL_PARAMS["F"]
    torch.testing.assert_close(api.dict_to_params(api.params_to_dict(params)), params)


def test_calibrate_recovers_a_simulated_beat(bundle):
    target = api.simulate(api.preset("V")).values[0]
    fitted = api.calibrate(target, steps=600)
    assert fitted["r2"] > 0.99
    np.testing.assert_allclose(api.simulate(fitted["params"], fitted["z0"]).values[0], target,
                               atol=0.004)  # fmt: skip


def test_real_examples_are_real_validation_beats(bundle):
    beats = api.real_examples("F", n=4, random_state=1)
    assert beats.values.shape == (4, 216)
    assert not beats.synthetic and beats.source == "real"
    assert beats.info["split"] == "validation" and "MIT-BIH" in beats.info["attribution"]


def test_missing_bundle_gives_a_clear_error(tmp_path):
    api.set_model_dir(tmp_path / "nowhere")
    with pytest.raises(FileNotFoundError, match="set_model_dir"):
        api.available_models()


def test_invalid_arguments_are_rejected(bundle):
    with pytest.raises(ValueError, match="beat_type"):
        api.generate("N")
    with pytest.raises(ValueError, match="model"):
        api.generate("V", model="vae")


def test_a_seed_not_in_the_bundle_names_the_available_seeds(bundle):
    with pytest.raises(ValueError, match="seed 7; available seeds: 0") as error:
        api.generate("V", seed=7)
    assert str(bundle) not in str(error.value)
    assert api.generate("V", 2, seed=0).values.shape == (2, 216)
