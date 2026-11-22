import pytest

from simheart.data.splits import DS1, DS2, PACED, build_splits, patient_id, patient_overlap


def test_standard_lists():
    assert len(DS1) == 22 and len(DS2) == 22
    assert not set(DS1) & set(DS2)
    assert not set(PACED) & (set(DS1) | set(DS2))


def test_standard_split_reports_the_201_202_patient_overlap():
    splits = build_splits()
    assert splits["train"] == list(DS1)
    assert splits["test"] == list(DS2)
    assert patient_id("202") == "201"
    assert patient_overlap(splits) == {"201"}


def test_strict_split_shares_no_patient():
    splits = build_splits(strict_patients=True)
    assert "202" not in splits["test"]
    assert patient_overlap(splits) == set()


def test_validation_records_move_out_of_training():
    splits = build_splits(validation_records=["101", "203"])
    assert splits["validation"] == ["101", "203"]
    assert "101" not in splits["train"] and "203" not in splits["train"]
    records = [r for split in splits.values() for r in split]
    assert len(records) == len(set(records))


def test_validation_records_must_come_from_ds1():
    with pytest.raises(ValueError, match="DS1"):
        build_splits(validation_records=["100"])
