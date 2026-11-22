"""The heartbeat classifier of Kachuee et al. (2018), as described in the SimGANs paper.

One convolution, five residual blocks (each: conv -> ReLU -> conv -> skip -> ReLU ->
max-pool), then two fully connected layers of 32 units. All convolutions have 32 kernels
of size 5. Kachuee et al. used 187-sample beats at 125 Hz; here the input is our
216-sample beats at 360 Hz (only the input length differs). The output is one logit: the
classifier answers "is this beat of class c?" (one-vs-rest, as in the SimGANs paper).

Sequence length through the blocks: 216 -> 106 -> 51 -> 24 -> 10 -> 3.
"""

import torch
from torch import nn

CHANNELS = 32
KERNEL = 5
N_BLOCKS = 5


class ResidualBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(CHANNELS, CHANNELS, KERNEL, padding=KERNEL // 2)
        self.conv2 = nn.Conv1d(CHANNELS, CHANNELS, KERNEL, padding=KERNEL // 2)
        self.pool = nn.MaxPool1d(kernel_size=5, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv2(torch.relu(self.conv1(x)))
        return self.pool(torch.relu(out + x))


class KachueeResNet(nn.Module):
    """Beat [B, 216] or [B, 1, 216] -> logit [B] (positive means "class c")."""

    def __init__(self, length: int = 216) -> None:
        super().__init__()
        self.stem = nn.Conv1d(1, CHANNELS, KERNEL, padding=KERNEL // 2)
        self.blocks = nn.Sequential(*[ResidualBlock() for _ in range(N_BLOCKS)])
        for _ in range(N_BLOCKS):
            length = (length - 5) // 2 + 1
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(CHANNELS * length, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, beats: torch.Tensor) -> torch.Tensor:
        x = self.stem(beats.reshape(len(beats), 1, -1))
        return self.head(self.blocks(x)).view(-1)
