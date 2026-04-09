"""Generic train / eval helpers for PyTorch.

Assumes ``DataLoader`` yields ``(x, y)`` batches. Classification mode expects
``[N, C]`` logits and ``y`` of shape ``[N]`` (``CrossEntropyLoss``-style).
Regression expects ``output`` and ``target`` broadcastable of equal shape.

**Loss aggregation**: metrics assume the criterion uses ``reduction="mean"`` over
the batch (PyTorch default for ``CrossEntropyLoss``, ``MSELoss``, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt


def _loader_len(loader: DataLoader) -> int | None:
    try:
        return len(loader)
    except TypeError:
        return None


def _weighted_loss_sum(loss: torch.Tensor, batch_size: int) -> float:
    """Turn batch-mean loss into sum of per-sample losses (for epoch average)."""
    return float(loss.item()) * batch_size


@dataclass
class TrainArgs:
    criterion: nn.Module
    optimizer: optim.Optimizer
    num_epochs: int
    device: torch.device = field(
        default_factory=lambda: torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    )


@dataclass
class EvalArgs:
    criterion: nn.Module = field(default_factory=nn.CrossEntropyLoss)
    device: torch.device = field(
        default_factory=lambda: torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    )


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    train_args: TrainArgs,
    test_loader: DataLoader | None = None,
    mode: str = "cla",
    debug_mode: int = 0,
    model_save_path: str | Path | None = None,
) -> tuple[nn.Module, pd.DataFrame]:
    """Train for ``num_epochs``; optionally evaluate on ``test_loader`` each epoch.

    ``mode``:
        - ``"cla"``: ``argmax`` on dim 1 vs ``y``; history column ``train_metric`` is accuracy.
        - ``"reg"``: SSE over batches; ``train_metric`` is **MSE** (mean squared error).

    ``debug_mode``: 0 silent, 1 train-only prints, 2 train+test prints (needs ``test_loader``).

    **Note**: call ``optimizer.zero_grad()`` each step — included here.
    """
    if mode not in ("cla", "reg"):
        raise ValueError("mode must be 'cla' or 'reg'")
    if debug_mode not in (0, 1, 2):
        raise ValueError("debug_mode must be 0, 1 or 2")
    if test_loader is None and debug_mode == 2:
        debug_mode = 1

    device = train_args.device
    nb = device.type == "cuda"

    total_train_losses: list[float] = []
    total_train_metrics: list[float] = []
    total_train_sampleses: list[int] = []
    total_test_losses: list[float] = []
    total_test_metrics: list[float] = []
    total_test_sampleses: list[int] = []

    for epoch in range(train_args.num_epochs):
        model.train()
        run_loss = 0.0
        run_metric_num = 0.0  # correct count (cla) or SSE (reg)
        n_train = 0

        pbar = tqdm(
            train_loader,
            total=_loader_len(train_loader),
            desc=f"Epoch {epoch + 1}/{train_args.num_epochs}",
        )
        for data, target in pbar:
            data = data.to(device, non_blocking=nb)
            target = target.to(device, non_blocking=nb)
            train_args.optimizer.zero_grad(set_to_none=True)
            output = model(data)
            loss = train_args.criterion(output, target)
            loss.backward()
            train_args.optimizer.step()

            bs = data.size(0)
            run_loss += _weighted_loss_sum(loss, bs)
            n_train += bs
            if mode == "cla":
                run_metric_num += (output.argmax(dim=1) == target).sum().item()
            else:
                run_metric_num += (output.float() - target.float()).pow(2).sum().item()

        if n_train == 0:
            raise RuntimeError("train_loader produced zero samples")

        epoch_train_loss = run_loss / n_train
        if mode == "cla":
            epoch_train_metric = run_metric_num / n_train
        else:
            epoch_train_metric = run_metric_num / n_train  # MSE

        total_train_losses.append(epoch_train_loss)
        total_train_metrics.append(epoch_train_metric)
        total_train_sampleses.append(n_train)

        epoch_test_loss = float("nan")
        epoch_test_metric = float("nan")
        n_test = 0

        if test_loader is not None:
            model.eval()
            te_loss = 0.0
            te_metric_num = 0.0
            with torch.no_grad():
                for data, target in test_loader:
                    data = data.to(device, non_blocking=nb)
                    target = target.to(device, non_blocking=nb)
                    output = model(data)
                    loss = train_args.criterion(output, target)
                    bs = data.size(0)
                    te_loss += _weighted_loss_sum(loss, bs)
                    n_test += bs
                    if mode == "cla":
                        te_metric_num += (output.argmax(dim=1) == target).sum().item()
                    else:
                        te_metric_num += (output.float() - target.float()).pow(2).sum().item()

            if n_test == 0:
                raise RuntimeError("test_loader produced zero samples")
            epoch_test_loss = te_loss / n_test
            if mode == "cla":
                epoch_test_metric = te_metric_num / n_test
            else:
                epoch_test_metric = te_metric_num / n_test

            total_test_losses.append(epoch_test_loss)
            total_test_metrics.append(epoch_test_metric)
            total_test_sampleses.append(n_test)
            model.train()

        if debug_mode == 1:
            if mode == "cla":
                print(
                    f"Epoch {epoch + 1}/{train_args.num_epochs} | "
                    f"train loss {epoch_train_loss:.4f} | acc {epoch_train_metric:.4f}"
                )
            else:
                print(
                    f"Epoch {epoch + 1}/{train_args.num_epochs} | "
                    f"train loss {epoch_train_loss:.4f} | RMSE {epoch_train_metric ** 0.5:.4f}"
                )

        if debug_mode == 2 and test_loader is not None:
            if mode == "cla":
                print(
                    f"Epoch {epoch + 1}/{train_args.num_epochs} | "
                    f"train loss {epoch_train_loss:.4f} | acc {epoch_train_metric:.4f} | "
                    f"test loss {epoch_test_loss:.4f} | acc {epoch_test_metric:.4f}"
                )
            else:
                print(
                    f"Epoch {epoch + 1}/{train_args.num_epochs} | "
                    f"train loss {epoch_train_loss:.4f} | RMSE {epoch_train_metric ** 0.5:.4f} | "
                    f"test loss {epoch_test_loss:.4f} | RMSE {epoch_test_metric ** 0.5:.4f}"
                )

    if model_save_path is not None:
        path = Path(model_save_path)
        path.mkdir(parents=True, exist_ok=True)
        out = path / f"model_epoch_{train_args.num_epochs}.pth"
        torch.save(model.state_dict(), out)

    rows: dict[str, Any] = {
        "train_loss": total_train_losses,
        "train_metric": total_train_metrics,
        "train_samples": total_train_sampleses,
    }
    if test_loader is not None:
        rows["test_loss"] = total_test_losses
        rows["test_metric"] = total_test_metrics
        rows["test_samples"] = total_test_sampleses

    return model, pd.DataFrame(rows)


def eval_model(
    model: nn.Module,
    test_loader: DataLoader,
    eval_args: EvalArgs,
    mode: str = "cla",
) -> tuple[float, float, int]:
    """Returns ``(avg_loss, metric, n_samples)``.

    - ``cla``: ``metric`` is accuracy in ``[0, 1]``.
    - ``reg``: ``metric`` is **RMSE** (root mean squared error).
    """
    if mode not in ("cla", "reg"):
        raise ValueError("mode must be 'cla' or 'reg'")

    device = eval_args.device
    nb = device.type == "cuda"
    model.eval()
    run_loss = 0.0
    metric_num = 0.0
    n_total = 0

    with torch.no_grad():
        for data, target in tqdm(
            test_loader,
            total=_loader_len(test_loader),
            desc="eval",
        ):
            data = data.to(device, non_blocking=nb)
            target = target.to(device, non_blocking=nb)
            output = model(data)
            loss = eval_args.criterion(output, target)
            bs = data.size(0)
            run_loss += _weighted_loss_sum(loss, bs)
            n_total += bs
            if mode == "cla":
                metric_num += (output.argmax(dim=1) == target).sum().item()
            else:
                metric_num += (output.float() - target.float()).pow(2).sum().item()

    if n_total == 0:
        raise RuntimeError("test_loader produced zero samples")

    avg_loss = run_loss / n_total
    if mode == "cla":
        metric = metric_num / n_total
    else:
        metric = (metric_num / n_total) ** 0.5

    return avg_loss, metric, n_total


# Backward-compatible names (older code used lowercase class names)
train_args = TrainArgs
eval_args = EvalArgs


def plot_losses_curve(res:pd.DataFrame, save_path:str=None, mode='cla'):
    """
    Plot the losses curve.
    Args:
        res: The result dataframe.
        save_path: The save path.
    """
    if mode == 'cla':
        y_label = 'accuracy'
        y_data = res['train_metric']
        y_data_test = res['test_metric']
    else:
        y_label = 'rmse'
        y_data = res['train_metric']
        y_data_test = res['test_metric']
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # Accuracy or Rmse
    axes[0].plot(y_data, label='train ' + y_label)
    axes[0].plot(y_data_test, label='test ' + y_label)
    axes[0].set_xlabel('epoch')
    axes[0].set_ylabel(y_label)
    axes[0].grid(True, linestyle='--', alpha=0.45)
    axes[0].legend()
    # Loss
    axes[1].plot(res['train_loss'], label='train loss')
    axes[1].plot(res['test_loss'], label='test loss')
    axes[1].set_xlabel('epoch')
    axes[1].set_ylabel('loss')
    axes[1].grid(True, linestyle='--', alpha=0.45)
    axes[1].legend()

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path)
        plt.close(fig)
    else:
        plt.show()
    

__all__ = [
    "TrainArgs",
    "EvalArgs",
    "train_args",
    "eval_args",
    "train_model",
    "eval_model",
    "plot_losses_curve",
]
