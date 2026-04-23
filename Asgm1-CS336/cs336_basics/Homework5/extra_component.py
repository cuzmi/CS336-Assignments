"""
Data loading, checkpointing, and a lightweight training loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn


def get_batch(
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    indices = torch.randint(0, len(dataset) - context_length, size=(batch_size,)).tolist()
    x = torch.stack(
        [torch.tensor(dataset[i : i + context_length], dtype=torch.long) for i in indices],
        dim=0,
    ).to(device)
    y = torch.stack(
        [torch.tensor(dataset[i + 1 : i + context_length + 1], dtype=torch.long) for i in indices],
        dim=0,
    ).to(device)
    return x, y


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str,
) -> None:
    checkpoint = {
        "model_dict": model.state_dict(),
        "optim_dict": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, out)


def load_checkpoint(
    src: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model_dict"])
    optimizer.load_state_dict(checkpoint["optim_dict"])
    return checkpoint["iteration"]


@dataclass
class TrainConfig:
    train_data: np.ndarray
    val_data: np.ndarray | None = None
    batch_size: int = 16
    context_length: int = 128
    num_steps: int = 1000
    device: str = "cpu"
    log_interval: int = 100
    eval_interval: int = 500
    eval_batches: int = 10
    save_interval: int = 1000
    out: str | None = None
    src: str | None = None


def _compute_loss(
    model: nn.Module,
    loss_fn: nn.Module | Any,
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    logits = model(x)
    if logits.ndim >= 3:
        vocab_size = logits.shape[-1]
        return loss_fn(logits.reshape(-1, vocab_size), y.reshape(-1))
    return loss_fn(logits, y)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataset: np.ndarray,
    loss_fn: nn.Module | Any,
    config: TrainConfig,
) -> float:
    was_training = model.training
    model.eval()

    losses = []
    for _ in range(config.eval_batches):
        x, y = get_batch(dataset, config.batch_size, config.context_length, config.device)
        loss = _compute_loss(model, loss_fn, x, y)
        losses.append(loss.item())

    if was_training:
        model.train()

    return float(sum(losses) / len(losses))


def train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module | Any,
    config: TrainConfig,
) -> dict[str, list[dict[str, float | int]]]:
    start_step = 0
    if config.src is not None:
        start_step = load_checkpoint(config.src, model, optimizer)

    model.to(config.device)
    model.train()

    history: dict[str, list[dict[str, float | int]]] = {
        "train": [],
        "val": [],
    }

    for step in range(start_step, config.num_steps):
        optimizer.zero_grad()
        x, y = get_batch(config.train_data, config.batch_size, config.context_length, config.device)
        loss = _compute_loss(model, loss_fn, x, y)
        loss.backward()
        optimizer.step()

        current_step = step + 1

        if current_step % config.log_interval == 0 or current_step == 1:
            train_loss = float(loss.item())
            history["train"].append({"step": current_step, "loss": train_loss})
            print(f"step={current_step} train_loss={train_loss:.4f}")

        if (
            config.val_data is not None
            and config.eval_interval > 0
            and current_step % config.eval_interval == 0
        ):
            val_loss = evaluate(model, config.val_data, loss_fn, config)
            history["val"].append({"step": current_step, "loss": val_loss})
            print(f"step={current_step} val_loss={val_loss:.4f}")

        if (
            config.out is not None
            and config.save_interval > 0
            and current_step % config.save_interval == 0
        ):
            save_checkpoint(model, optimizer, current_step, config.out)

    return history
