"""Amplitude scaling of heartbeats.

Beats are stored in millivolts. For GAN training they are rescaled, one beat at a time,
into the amplitude range of the ECGSYN simulator's output, so that the Euler loss
(which compares a beat with the simulator's equations) works on the same scale as the
real data. The range comes from the SimGANs authors' code (``DreamStudioAI/sim_gan``,
``ecg_dataset_pytorch.scale_signal``). Whether to keep it is revisited when the
simulator is built.
"""

import numpy as np

SIM_MIN = -0.01563
SIM_SPAN = 0.042557  # SIM_MIN + SIM_SPAN = 0.026927 is the top of the range


def minmax_to_sim_range(beats: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Rescale each beat (last axis) so its minimum is SIM_MIN and its maximum is the top."""
    low = beats.min(axis=-1, keepdims=True)
    high = beats.max(axis=-1, keepdims=True)
    return ((beats - low) / np.maximum(high - low, eps) * SIM_SPAN + SIM_MIN).astype(beats.dtype)
