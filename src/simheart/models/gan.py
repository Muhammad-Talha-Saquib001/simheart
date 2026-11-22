"""DCGAN generator and discriminator for single 216-sample heartbeats.

The layer settings follow the SimGANs authors' code (``DreamStudioAI/sim_gan``,
``gan_models/models/sim_gan_euler.py``), which settles the paper's ambiguity about the
number of generator layers. Sequence lengths:

* generator:     1 -> 4 -> 7 -> 14 -> 27 -> 54 -> 108 -> 216   (7 transposed convolutions;
                 the single kernel-3 layer is what makes the output exactly 216 long)
* discriminator: 216 -> 108 -> 54 -> 27 -> 13 -> 6 -> 1        (6 convolutions)

Layers followed by batch normalisation have no bias (the normalisation would cancel it).
The same networks are used for the plain DCGAN and for SimDCGAN; only the training loss
differs.
"""

import torch
from torch import nn

NOISE_DIM = 100
BEAT_LENGTH = 216

# (in_channels, out_channels, kernel, stride, padding)
_GENERATOR_LAYERS = [
    (NOISE_DIM, 2048, 4, 1, 0),
    (2048, 1024, 4, 1, 0),
    (1024, 512, 4, 2, 1),
    (512, 256, 3, 2, 1),
    (256, 128, 4, 2, 1),
    (128, 64, 4, 2, 1),
    (64, 1, 4, 2, 1),
]
_DISCRIMINATOR_LAYERS = [
    (1, 64, 4, 2, 1),
    (64, 128, 4, 2, 1),
    (128, 256, 4, 2, 1),
    (256, 512, 4, 2, 1),
    (512, 1024, 4, 2, 1),
    (1024, 1, 5, 2, 0),
]


class Generator(nn.Module):
    """Noise [B, 100] -> heartbeat [B, 1, 216] (no output activation)."""

    def __init__(self) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last = len(_GENERATOR_LAYERS) - 1
        for i, (c_in, c_out, k, s, p) in enumerate(_GENERATOR_LAYERS):
            layers.append(nn.ConvTranspose1d(c_in, c_out, k, s, p, bias=i == last))
            if i != last:
                layers += [nn.BatchNorm1d(c_out), nn.ReLU(inplace=True)]
        self.net = nn.Sequential(*layers)

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        return self.net(noise.view(-1, NOISE_DIM, 1))


class Discriminator(nn.Module):
    """Heartbeat [B, 1, 216] -> logit [B] (real if positive)."""

    def __init__(self) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last = len(_DISCRIMINATOR_LAYERS) - 1
        for i, (c_in, c_out, k, s, p) in enumerate(_DISCRIMINATOR_LAYERS):
            layers.append(nn.Conv1d(c_in, c_out, k, s, p, bias=False))
            if 0 < i < last:
                layers.append(nn.BatchNorm1d(c_out))
            if i != last:
                layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, beats: torch.Tensor) -> torch.Tensor:
        return self.net(beats.view(-1, 1, BEAT_LENGTH)).view(-1)


def init_weights(module: nn.Module) -> None:
    """DCGAN initialisation: convolutions N(0, 0.02), batch-norm scale N(1, 0.02), shift 0."""
    if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)):
        nn.init.normal_(module.weight, 0.0, 0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm1d):
        nn.init.normal_(module.weight, 1.0, 0.02)
        nn.init.zeros_(module.bias)
