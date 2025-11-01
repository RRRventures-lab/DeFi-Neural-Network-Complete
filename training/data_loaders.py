"""
Data Loading Utilities for Training

Handles:
- Walk-forward validation splitting
- Batch creation and loading
- Data preprocessing and normalization
- Train/val/test splits
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)


class TimeSeriesDataset(Dataset):
    """
    PyTorch Dataset for time-series windows.

    Expects pre-computed windows from feature pipeline.
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        normalize: bool = False,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None
    ):
        """
        Initialize dataset.

        Args:
            X: Feature windows of shape (num_samples, timesteps, features)
            y: Target values of shape (num_samples,) or (num_samples, 1)
            normalize: Whether to normalize features
            mean: Mean for normalization (computed if None)
            std: Std for normalization (computed if None)
        """
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

        # Handle y shape
        if self.y.ndim == 1:
            self.y = self.y.unsqueeze(1)

        # Normalization
        self.normalize = normalize
        if normalize:
            if mean is None:
                mean = X.mean(axis=(0, 1), keepdims=True)
            if std is None:
                std = X.std(axis=(0, 1), keepdims=True)
                std[std == 0] = 1.0  # Avoid division by zero

            self.mean = torch.from_numpy(mean).float()
            self.std = torch.from_numpy(std).float()

            # Normalize X
            self.X = (self.X - self.mean) / self.std
        else:
            self.mean = None
            self.std = None

        assert len(self.X) == len(self.y), "X and y must have same number of samples"

    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get single sample."""
        return self.X[idx], self.y[idx]


class WalkForwardValidator:
    """
    Walk-forward validation splitter for time-series.

    Ensures temporal order is preserved (no look-ahead bias).
    """

    def __init__(
        self,
        num_windows: int,
        validation_size: float = 0.2,
        num_steps: int = 5
    ):
        """
        Initialize walk-forward validator.

        Args:
            num_windows: Total number of data windows
            validation_size: Fraction of data for each validation step
            num_steps: Number of walk-forward steps
        """
        self.num_windows = num_windows
        self.validation_size = validation_size
        self.num_steps = num_steps

        # Calculate sizes
        self.val_size = int(num_windows * validation_size)
        self.train_size = num_windows - self.val_size

    def get_splits(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Get walk-forward train/val splits.

        Returns:
            List of (train_indices, val_indices) tuples
        """
        splits = []

        # Step size: move validation window forward
        step_size = self.val_size

        for step in range(self.num_steps):
            val_start = step * step_size
            val_end = val_start + self.val_size

            if val_end > self.num_windows:
                break

            train_indices = np.arange(0, val_start)
            val_indices = np.arange(val_start, val_end)

            splits.append((train_indices, val_indices))

        return splits


def create_data_loaders(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 32,
    validation_split: float = 0.2,
    shuffle_train: bool = True,
    normalize: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation data loaders.

    Args:
        X: Feature windows (num_samples, timesteps, features)
        y: Target values (num_samples,)
        batch_size: Batch size
        validation_split: Fraction for validation
        shuffle_train: Whether to shuffle training data
        normalize: Whether to normalize features

    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Compute normalization statistics on training data only
    num_samples = len(X)
    val_size = int(num_samples * validation_split)
    train_size = num_samples - val_size

    train_indices = np.arange(0, train_size)
    val_indices = np.arange(train_size, num_samples)

    X_train = X[train_indices]
    y_train = y[train_indices]
    X_val = X[val_indices]
    y_val = y[val_indices]

    # Compute normalization on training data only (prevent data leakage)
    if normalize:
        train_mean = X_train.mean(axis=(0, 1), keepdims=True)
        train_std = X_train.std(axis=(0, 1), keepdims=True)
        train_std[train_std == 0] = 1.0
    else:
        train_mean = None
        train_std = None

    # Create datasets
    train_dataset = TimeSeriesDataset(
        X_train, y_train,
        normalize=normalize,
        mean=train_mean,
        std=train_std
    )

    val_dataset = TimeSeriesDataset(
        X_val, y_val,
        normalize=normalize,
        mean=train_mean,
        std=train_std
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        drop_last=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False
    )

    logger.info(f"Train set: {len(train_dataset)} samples")
    logger.info(f"Validation set: {len(val_dataset)} samples")

    return train_loader, val_loader


def create_walk_forward_loaders(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 32,
    validation_fraction: float = 0.2,
    num_steps: int = 5,
    normalize: bool = True
) -> List[Tuple[DataLoader, DataLoader]]:
    """
    Create walk-forward validation data loaders.

    Ensures no look-ahead bias in time-series.

    Args:
        X: Feature windows
        y: Target values
        batch_size: Batch size
        validation_fraction: Fraction for each validation window
        num_steps: Number of walk-forward steps
        normalize: Whether to normalize features

    Returns:
        List of (train_loader, val_loader) tuples for each step
    """
    validator = WalkForwardValidator(
        num_windows=len(X),
        validation_size=validation_fraction,
        num_steps=num_steps
    )

    splits = validator.get_splits()
    loaders = []

    for step, (train_idx, val_idx) in enumerate(splits):
        X_train = X[train_idx]
        y_train = y[train_idx]
        X_val = X[val_idx]
        y_val = y[val_idx]

        # Compute normalization on training data
        if normalize:
            train_mean = X_train.mean(axis=(0, 1), keepdims=True)
            train_std = X_train.std(axis=(0, 1), keepdims=True)
            train_std[train_std == 0] = 1.0
        else:
            train_mean = None
            train_std = None

        # Create datasets
        train_dataset = TimeSeriesDataset(
            X_train, y_train,
            normalize=normalize,
            mean=train_mean,
            std=train_std
        )

        val_dataset = TimeSeriesDataset(
            X_val, y_val,
            normalize=normalize,
            mean=train_mean,
            std=train_std
        )

        # Create loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False
        )

        loaders.append((train_loader, val_loader))

        logger.info(
            f"Step {step + 1}: Train={len(train_dataset)}, "
            f"Val={len(val_dataset)}"
        )

    return loaders


def prepare_data(
    X_windows: List[np.ndarray],
    y_targets: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare data from Stage 2 output.

    Args:
        X_windows: List of window arrays
        y_targets: Array of target values

    Returns:
        Tuple of (X_combined, y_array)
    """
    # Convert list of windows to numpy array
    if isinstance(X_windows, list):
        X = np.array(X_windows)
    else:
        X = X_windows

    # Ensure targets are correct shape
    y = np.array(y_targets).flatten()

    assert len(X) == len(y), "X and y must have same number of samples"

    logger.info(f"Data prepared: X shape {X.shape}, y shape {y.shape}")

    return X, y
