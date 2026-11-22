"""Patient-level train / validation / test splits for MIT-BIH.

The test split is DS2 and the training split is DS1 from de Chazal et al. (2004),
the standard inter-patient benchmark. Every record of a patient goes into one split,
so the test set measures performance on unseen people. The four paced records are
left out.

One known flaw: records 201 and 202 come from the same patient but sit in DS1 and DS2
respectively. ``strict_patients=True`` removes 202 from the test set to close that gap.
"""

DS1: tuple[str, ...] = (
    "101", "106", "108", "109", "112", "114", "115", "116", "118", "119", "122",
    "124", "201", "203", "205", "207", "208", "209", "215", "220", "223", "230",
)  # fmt: skip
DS2: tuple[str, ...] = (
    "100", "103", "105", "111", "113", "117", "121", "123", "200", "202", "210",
    "212", "213", "214", "219", "221", "222", "228", "231", "232", "233", "234",
)  # fmt: skip
PACED: tuple[str, ...] = ("102", "104", "107", "217")

# Records that belong to the same patient as another record (record -> patient ID).
SAME_PATIENT: dict[str, str] = {"202": "201"}

SPLIT_NAMES: tuple[str, ...] = ("train", "validation", "test")


def patient_id(record: str) -> str:
    """Return the patient ID for a record. Most records are their own patient."""
    return SAME_PATIENT.get(record, record)


def build_splits(
    validation_records: list[str] | tuple[str, ...] = (),
    strict_patients: bool = False,
) -> dict[str, list[str]]:
    """Return the record lists for the train, validation and test splits.

    Validation records are taken out of DS1. With ``strict_patients`` the test split
    drops any record whose patient also appears in the training or validation data.
    """
    validation = list(dict.fromkeys(validation_records))
    unknown = [r for r in validation if r not in DS1]
    if unknown:
        raise ValueError(f"Validation records must come from DS1; not in DS1: {unknown}")

    train = [r for r in DS1 if r not in validation]
    test = list(DS2)
    if strict_patients:
        development_patients = {patient_id(r) for r in DS1}
        test = [r for r in test if patient_id(r) not in development_patients]
    return {"train": train, "validation": validation, "test": test}


def patient_overlap(splits: dict[str, list[str]]) -> set[str]:
    """Return the patient IDs that appear in more than one split."""
    seen: dict[str, str] = {}
    shared: set[str] = set()
    for split_name, records in splits.items():
        for record in records:
            patient = patient_id(record)
            if patient in seen and seen[patient] != split_name:
                shared.add(patient)
            seen.setdefault(patient, split_name)
    return shared
