"""Amplitude scaling of heartbeats.

Beats are stored in millivolts. For GAN training they are rescaled, one beat at a time,
into the amplitude range of the ECGSYN simulator's output, so that the Euler loss
(which compares a beat with the simulator's equations) works on the same scale as the
real data.

The range [SIM_MIN, SIM_MAX] = [-0.01563, 0.042557] comes from the SimGANs authors' code
(``DreamStudioAI/sim_gan``, ``ecg_dataset_pytorch.scale_signal(min_val, max_val)``). Their
function multiplies by ``max_val`` instead of ``max_val - min_val``, so it actually maps
beats to [-0.01563, 0.026927]. We use the intended range: it matches the simulator's
output (typical beats span about -0.016 to 0.046) and puts the flat part of real normal
beats at the simulator's baseline of 0 (checked with scripts/plot_simulator.py).
"""

import numpy as np

SIM_MIN = -0.01563
SIM_MAX = 0.042557
SIM_SPAN = SIM_MAX - SIM_MIN


def sim_to_unit(beats):
    """Map the simulator range [SIM_MIN, SIM_MAX] linearly onto [-1, 1] (NumPy or torch).

    The GANs work on this scale; the Euler loss maps back with ``unit_to_sim``.
    """
    return (beats - SIM_MIN) / SIM_SPAN * 2 - 1


def unit_to_sim(beats):
    """Inverse of ``sim_to_unit``: [-1, 1] back to simulator units (NumPy or torch)."""
    return (beats + 1) / 2 * SIM_SPAN + SIM_MIN


def minmax_to_sim_range(beats: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Rescale each beat (last axis) so its minimum is SIM_MIN and its maximum is SIM_MAX."""
    low = beats.min(axis=-1, keepdims=True)
    high = beats.max(axis=-1, keepdims=True)
    return ((beats - low) / np.maximum(high - low, eps) * SIM_SPAN + SIM_MIN).astype(beats.dtype)
