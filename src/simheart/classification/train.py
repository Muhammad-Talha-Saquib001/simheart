"""Training and scoring the one-vs-rest classifier."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from simheart.classification.model import KachueeResNet


def use_tuned_cudnn() -> None:
    """Let cuDNN time its (deterministic) convolution kernels and keep the fastest.

    For this small 1-D network, cuDNN's untuned default on an RTX 2060 (torch 2.15, cuDNN
    9.26) picks a kernel that needs 8.7 GB and 3.7 s per batch of 256; tuned, a batch takes
    about 7 ms. ``set_seed`` keeps ``cudnn.deterministic = True``, so only deterministic
    kernels are considered; which one wins can vary between machines, so results can differ
    in the last decimals across hardware.
    """
    torch.backends.cudnn.benchmark = True


@dataclass
class ClassifierConfig:
    base_epochs: int = 20
    finetune_epochs: int = 5
    lr: float = 1e-3
    batch_size: int = 256


def train_classifier(
    x: np.ndarray,
    y: np.ndarray,
    epochs: int,
    config: ClassifierConfig,
    device: torch.device,
    init_state: dict | None = None,
    seed: int = 0,
    on_epoch: Callable[[int, KachueeResNet], None] | None = None,
) -> tuple[dict, list[float]]:
    """Train (or, given ``init_state``, fine-tune) on beats ``x`` [N, 216] and 0/1 labels ``y``.

    Plain binary cross-entropy, Adam, shuffled mini-batches. Returns the final weights (on
    the CPU) and the mean loss per epoch. ``on_epoch(epoch, model)`` is called after each
    epoch (the pilot uses it to score validation beats).
    """
    use_tuned_cudnn()
    generator = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    model = KachueeResNet().to(device)
    if init_state is not None:
        model.load_state_dict(init_state)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    loss_fn = nn.BCEWithLogitsLoss()
    beats = torch.as_tensor(x, dtype=torch.float32)
    labels = torch.as_tensor(y, dtype=torch.float32)

    losses = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(beats), generator=generator)
        total = 0.0
        for start in range(0, len(order), config.batch_size):
            index = order[start : start + config.batch_size]
            optimizer.zero_grad()
            loss = loss_fn(model(beats[index].to(device)), labels[index].to(device))
            loss.backward()
            optimizer.step()
            total += loss.item() * len(index)
        losses.append(total / len(beats))
        if on_epoch is not None:
            on_epoch(epoch, model)
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return state, losses


@torch.no_grad()
def predict_scores(
    model_or_state: KachueeResNet | dict,
    x: np.ndarray,
    device: torch.device,
    batch_size: int = 4096,
) -> np.ndarray:
    """Probability that each beat belongs to the positive class: [N]."""
    use_tuned_cudnn()
    if isinstance(model_or_state, dict):
        model = KachueeResNet().to(device)
        model.load_state_dict(model_or_state)
    else:
        model = model_or_state
    was_training = model.training
    model.eval()
    beats = torch.as_tensor(x, dtype=torch.float32)
    scores = [
        torch.sigmoid(model(beats[i : i + batch_size].to(device))).cpu()
        for i in range(0, len(beats), batch_size)
    ]
    model.train(was_training)
    return torch.cat(scores).numpy()
