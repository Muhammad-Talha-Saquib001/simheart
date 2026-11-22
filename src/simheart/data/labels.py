"""Mapping from MIT-BIH annotation symbols to the AAMI heartbeat classes.

MIT-BIH annotation files contain two kinds of symbols:

* beat annotations, which mark an R-peak and say what kind of beat it is, and
* non-beat annotations, which mark events such as rhythm changes (``+``), noise (``~``)
  or ventricular flutter (``[``, ``!``, ``]``).

Only beat annotations become training examples. Following the AAMI EC57 grouping,
beats map to N, S, V and F; the Q group (paced and unclassifiable beats) is excluded
from this project. Non-beat symbols are never turned into beats.
"""

CLASSES: tuple[str, ...] = ("N", "S", "V", "F")
CLASS_TO_INDEX: dict[str, int] = {name: i for i, name in enumerate(CLASSES)}

SYMBOL_TO_CLASS: dict[str, str] = {
    # N: normal and bundle branch block beats, atrial and nodal escape beats
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
    # S: supraventricular ectopic beats
    "A": "S", "a": "S", "J": "S", "S": "S",
    # V: ventricular ectopic beats
    "V": "V", "E": "V",
    # F: fusion of ventricular and normal beats
    "F": "F",
}  # fmt: skip

# AAMI class Q: paced, fusion of paced and normal, and unclassifiable beats.
EXCLUDED_BEAT_SYMBOLS: frozenset[str] = frozenset({"/", "f", "Q"})


def aami_class(symbol: str) -> str | None:
    """Return the project class (N, S, V or F) for a symbol, or None if it is not used."""
    return SYMBOL_TO_CLASS.get(symbol)


def is_beat_symbol(symbol: str) -> bool:
    """Return True if the symbol marks a heartbeat (used or excluded), False otherwise."""
    return symbol in SYMBOL_TO_CLASS or symbol in EXCLUDED_BEAT_SYMBOLS
