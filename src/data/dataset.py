"""
dataset.py — Thin PyTorch Dataset wrapper for pre-built (X, Y) tensor pairs.

train_model() performs manual random-index sampling directly on tensors and
does not require a DataLoader.  This class is provided for completeness and
potential future use (e.g. DataLoader-based evaluation loops).
"""

import torch
from torch.utils.data import Dataset


class ReservoirDataset(Dataset):
    """
    Wraps (X, Y) tensor pairs produced by build_dataset().

    Parameters
    ----------
    X : (N, T*D) input tensor
    Y : (N, D)   target tensor
    """

    def __init__(self, X: torch.Tensor, Y: torch.Tensor):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]