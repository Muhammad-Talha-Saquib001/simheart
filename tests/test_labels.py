from simheart.data.labels import CLASSES, EXCLUDED_BEAT_SYMBOLS, aami_class, is_beat_symbol


def test_aami_grouping():
    expected = {
        "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
        "A": "S", "a": "S", "J": "S", "S": "S",
        "V": "V", "E": "V",
        "F": "F",
    }  # fmt: skip
    for symbol, beat_class in expected.items():
        assert aami_class(symbol) == beat_class
        assert is_beat_symbol(symbol)


def test_q_beats_are_beats_but_excluded():
    for symbol in EXCLUDED_BEAT_SYMBOLS:
        assert is_beat_symbol(symbol)
        assert aami_class(symbol) is None


def test_non_beat_annotations_are_never_beats():
    for symbol in ["+", "~", "|", '"', "x", "[", "]", "!"]:
        assert not is_beat_symbol(symbol)
        assert aami_class(symbol) is None


def test_class_order():
    assert CLASSES == ("N", "S", "V", "F")
