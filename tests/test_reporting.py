import pytest

from simheart.reporting import (
    FINETUNE_EPOCHS,
    choose_pilot_settings,
    default_seeds,
    markdown_table,
    mean_std,
    missing_runs,
)


def pilot_rows():
    """lr 1e-3 peaks higher on both classes (V at epoch 2, S at epoch 3)."""
    aps = {
        (0.001, "V"): [0.80, 0.90, 0.85],
        (0.001, "S"): [0.30, 0.40, 0.50],
        (0.0003, "V"): [0.70, 0.75, 0.80],
        (0.0003, "S"): [0.20, 0.30, 0.35],
    }
    return [
        {"class": c, "lr": lr, "epoch": epoch, "val_ap": ap}
        for (lr, c), values in aps.items()
        for epoch, ap in enumerate(values, start=1)
    ]


def test_pilot_rule_picks_the_better_rate_then_the_best_mean_epoch():
    choice = choose_pilot_settings(pilot_rows())
    assert choice["lr"] == 0.001
    # mean over V and S per epoch at lr 1e-3: 0.55, 0.65, 0.675 -> epoch 3
    assert choice["base_epochs"] == 3
    assert choice["finetune_epochs"] == FINETUNE_EPOCHS
    assert choice["evidence"]["mean_val_ap_at_chosen_epoch"] == pytest.approx(0.675)


def test_pilot_rule_prefers_fewer_epochs_on_ties():
    rows = [{"class": "V", "lr": 0.001, "epoch": e, "val_ap": 0.5} for e in (1, 2, 3)]
    assert choose_pilot_settings(rows)["base_epochs"] == 1


def test_default_seed_is_the_lowest_mmd_per_model_and_class():
    rows = [
        {"class": "V", "method": "dcgan", "seed": "0", "mmd": "0.12"},
        {"class": "V", "method": "dcgan", "seed": "1", "mmd": "0.09"},
        {"class": "V", "method": "simdcgan", "seed": "0", "mmd": "0.08"},
        {"class": "V", "method": "simdcgan", "seed": "2", "mmd": "0.10"},
        {"class": "V", "method": "simulator", "seed": "2", "mmd": "0.01"},
        {"class": "V", "method": "real", "seed": "", "mmd": "0.11"},
    ]
    assert default_seeds(rows) == {("dcgan", "V"): 1, ("simdcgan", "V"): 0}


def test_missing_runs_lists_unfinished_training(tmp_path):
    (tmp_path / "dcgan_s0_V").mkdir()
    (tmp_path / "dcgan_s0_V" / "summary.json").write_text("{}")
    (tmp_path / "simdcgan_s0_V").mkdir()  # started, not finished
    missing = missing_runs(tmp_path, classes=["V"], seeds=[0])
    assert missing == ["simdcgan_s0_V"]


def test_markdown_table_formats_numbers_and_text():
    table = markdown_table(
        [{"name": "V", "ap": 0.81234}, {"name": "F", "ap": 0.5}],
        [("name", "Class"), ("ap", "AP")],
        {"ap": ".3f"},
    )
    assert table.splitlines() == ["| Class | AP |", "|---|---|", "| V | 0.812 |", "| F | 0.500 |"]


def test_mean_std():
    assert mean_std([1.0, 3.0]) == (2.0, 1.0)
