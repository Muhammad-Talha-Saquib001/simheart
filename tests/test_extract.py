import numpy as np

from simheart.data.labels import CLASS_TO_INDEX
from simheart.data.mitbih import AT_EDGE, EXCLUDED_CLASS, NON_BEAT, extract_beats


def test_windows_are_centred_on_the_r_peak_and_annotations_are_accounted_for():
    signal = np.arange(1000, dtype=np.float32)
    samples = np.array([10, 300, 500, 600, 700, 900])
    symbols = ["N", "V", "/", "F", "+", "N"]

    result = extract_beats(signal, samples, symbols, samples_before=72, samples_after=144)

    assert result.beats.shape == (2, 216)
    assert result.r_peaks.tolist() == [300, 600]
    assert result.labels.tolist() == [CLASS_TO_INDEX["V"], CLASS_TO_INDEX["F"]]
    assert result.beats[0, 72] == signal[300]
    assert result.beats[0, 0] == signal[300 - 72]
    assert result.beats[0, -1] == signal[300 + 143]
    assert result.skipped == {
        ("N", AT_EDGE): 2,  # 10 is too close to the start, 900 too close to the end
        ("/", EXCLUDED_CLASS): 1,
        ("+", NON_BEAT): 1,
    }


def test_no_beats_gives_empty_array_with_right_width():
    result = extract_beats(np.zeros(500, dtype=np.float32), np.array([250]), ["+"])
    assert result.beats.shape == (0, 216)
    assert result.labels.shape == (0,)
