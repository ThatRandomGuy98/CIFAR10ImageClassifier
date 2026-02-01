# utils/test_utils.py

from __future__ import annotations

from typing import List, Tuple
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


def predict_all(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:

    model.eval()
    y_true: List[np.ndarray] = []
    y_pred: List[np.ndarray] = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)

            logits = model(imgs)
            preds = logits.argmax(dim=1).cpu().numpy()

            y_pred.append(preds)
            y_true.append(labels.cpu().numpy())

    return np.concatenate(y_true), np.concatenate(y_pred)


def test_accuracy(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    verbose: bool = True,
) -> float:

    y_true, y_pred = predict_all(model, test_loader, device)
    acc = (y_true == y_pred).mean() * 100.0

    if verbose:
        print(f"Test Accuracy: {acc:.2f}%")

    return float(acc)


def confusion_matrix_np(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int,
) -> np.ndarray:

    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def per_class_accuracy_from_cm(
    cm: np.ndarray,
    class_names: List[str],
) -> List[Tuple[str, float, int]]:

    support = cm.sum(axis=1)  # true count per class (row sums)
    acc = cm.diagonal() / (support + 1e-12)

    results: List[Tuple[str, float, int]] = []
    for name, a, s in zip(class_names, acc, support):
        results.append((name, float(a), int(s)))

    return results


# Backwards-compatible wrapper so main.py doesn't need changes
def test_loop(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> float:

    return test_accuracy(model=model, test_loader=test_loader, device=device, verbose=True)
