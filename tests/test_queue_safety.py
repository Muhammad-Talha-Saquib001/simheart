"""Crash-safety pieces of the unattended batch (Phases 14-16)."""

import importlib.util
import json
from pathlib import Path

import pytest

from simheart.data.synthetic import ALL_METHODS, EXTRA_METHODS, METHODS

ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cut_off_last_result_line_is_moved_aside_and_the_rest_kept(tmp_path):
    read_results = load_script("run_augmentation").read_results
    path = tmp_path / "results.jsonl"
    good = [json.dumps({"key": "a", "ap": 0.5}), json.dumps({"key": "b", "ap": 0.25})]
    path.write_text("\n".join(good) + '\n{"key": "c", "ap"')  # a crash mid-write

    rows = read_results(path)

    assert [r["key"] for r in rows] == ["a", "b"]
    assert path.read_text() == "\n".join(good) + "\n"  # byte-identical good lines
    assert (tmp_path / "results.jsonl.partial").read_text() == '{"key": "c", "ap"\n'
    assert read_results(path) == rows  # a second read changes nothing


def test_corruption_before_the_last_line_stops_instead_of_guessing(tmp_path):
    read_results = load_script("run_augmentation").read_results
    path = tmp_path / "results.jsonl"
    path.write_text('{"key": "a"}\nnot json\n{"key": "c"}\n')
    with pytest.raises(SystemExit, match="line 2"):
        read_results(path)


def test_missing_results_file_reads_as_empty(tmp_path):
    assert load_script("run_augmentation").read_results(tmp_path / "none.jsonl") == []


def test_batch_methods_are_extras_and_main_methods_are_unchanged():
    assert METHODS == ("simulator", "dcgan", "simdcgan")
    new = {"abl_weight_low", "abl_weight_high", "abl_best_of_k", "abl_gaussian", "dcgan_sn",
           "simdcgan_sn"}  # fmt: skip
    assert new <= set(EXTRA_METHODS) and set(ALL_METHODS) == set(METHODS) | set(EXTRA_METHODS)


def test_every_batch_model_has_a_config_and_the_queue_starts_at_phase_14():
    train_all = load_script("train_all")
    for model in ["abl_weight_low", "abl_weight_high", "abl_best_of_k", "abl_gaussian",
                  "dcgan_sn", "simdcgan_sn"]:  # fmt: skip
        assert (ROOT / train_all.MODELS[model]).exists()
    pipeline = load_script("reproduce_results")
    names = [d for d, _, _ in pipeline.STEPS]
    assert names.index(pipeline.FIRST_BATCH_STEP) < names.index("Report assets") == len(names) - 1


def test_queue_step_markers_have_stable_names():
    queue = load_script("run_queue")
    assert queue.slug("V ten seeds: evaluate generators") == "v-ten-seeds-evaluate-generators"
    mem = queue.meminfo()
    assert mem["available"] > 0 and mem["swap_used"] >= 0
