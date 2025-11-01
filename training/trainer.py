"""
Trainer Class for Neural Network Training

Handles:
- Training loop with batching
- Validation during training
- Early stopping mechanism
- Model checkpointing
- Learning rate scheduling
- Gradient clipping
- Comprehensive metrics tracking
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Callable
from datetime import datetime
import logging
import json

logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Early stopping mechanism to prevent overfitting.

    Monitors validation metric and stops training if no improvement.
    """

    def __init__(
        self,
        patience: int = 15,
        min_delta: float = 1e-4,
        metric: str = 'loss'
    ):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs with no improvement before stopping
            min_delta: Minimum change to qualify as improvement
            metric: Metric to monitor ('loss', 'mae', 'mape')
        """
        self.patience = patience
        self.min_delta = min_delta
        self.metric = metric
        self.counter = 0
        self.best_value = None
        self.early_stop = False

    def __call__(self, current_value: float) -> bool:
        """
        Check if early stopping criterion is met.

        Args:
            current_value: Current metric value

        Returns:
            True if should stop, False otherwise
        """
        if self.best_value is None:
            self.best_value = current_value
        elif current_value < self.best_value - self.min_delta:
            self.best_value = current_value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


class ModelCheckpoint:
    """
    Model checkpointing to save best models.

    Saves model when validation metric improves.
    """

    def __init__(
        self,
        dirpath: str = './checkpoints',
        filename: str = 'best_model.pt',
        monitor: str = 'val_loss',
        mode: str = 'min'
    ):
        """
        Initialize checkpoint saver.

        Args:
            dirpath: Directory to save checkpoints
            filename: Checkpoint filename
            monitor: Metric to monitor
            mode: 'min' for loss, 'max' for accuracy
        """
        self.dirpath = Path(dirpath)
        self.dirpath.mkdir(parents=True, exist_ok=True)
        self.filename = filename
        self.monitor = monitor
        self.mode = mode
        self.best_value = None

    def save(self, model: nn.Module, current_value: float) -> bool:
        """
        Save model if current metric is best.

        Args:
            model: Model to save
            current_value: Current metric value

        Returns:
            True if model was saved, False otherwise
        """
        if self.best_value is None:
            self.best_value = current_value
            self._save_model(model)
            return True

        is_improvement = (
            (self.mode == 'min' and current_value < self.best_value) or
            (self.mode == 'max' and current_value > self.best_value)
        )

        if is_improvement:
            self.best_value = current_value
            self._save_model(model)
            return True

        return False

    def _save_model(self, model: nn.Module):
        """Save model state dict."""
        filepath = self.dirpath / self.filename
        torch.save(model.state_dict(), filepath)
        logger.info(f"Model saved to {filepath}")

    def load_best(self, model: nn.Module) -> nn.Module:
        """Load best saved model."""
        filepath = self.dirpath / self.filename
        if filepath.exists():
            model.load_state_dict(torch.load(filepath))
            logger.info(f"Loaded best model from {filepath}")
        return model


class Trainer:
    """
    Neural network trainer with comprehensive training features.

    Handles training loops, validation, early stopping, and checkpointing.
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataloader,
        val_dataloader,
        loss_fn: nn.Module,
        optimizer: optim.Optimizer,
        device: str = 'cpu',
        scheduler = None,
        max_grad_norm: float = 1.0
    ):
        """
        Initialize trainer.

        Args:
            model: Neural network model
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            loss_fn: Loss function
            optimizer: Optimizer
            device: Device to train on
            scheduler: Learning rate scheduler
            max_grad_norm: Maximum gradient norm for clipping
        """
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.max_grad_norm = max_grad_norm

        # Training state
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }

        # Early stopping and checkpointing
        self.early_stopping = None
        self.checkpoint = None

    def set_early_stopping(
        self,
        patience: int = 15,
        min_delta: float = 1e-4
    ):
        """
        Configure early stopping.

        Args:
            patience: Epochs with no improvement before stopping
            min_delta: Minimum improvement threshold
        """
        self.early_stopping = EarlyStopping(
            patience=patience,
            min_delta=min_delta
        )

    def set_checkpoint(
        self,
        dirpath: str = './checkpoints',
        filename: str = 'best_model.pt'
    ):
        """
        Configure model checkpointing.

        Args:
            dirpath: Directory for checkpoints
            filename: Checkpoint filename
        """
        self.checkpoint = ModelCheckpoint(
            dirpath=dirpath,
            filename=filename,
            monitor='val_loss',
            mode='min'
        )

    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.

        Returns:
            Dict with epoch metrics
        """
        self.model.train()
        total_loss = 0.0
        batch_count = 0

        for batch_idx, (X_batch, y_batch) in enumerate(self.train_dataloader):
            X_batch = X_batch.to(self.device).float()
            y_batch = y_batch.to(self.device).float()

            # Forward pass
            predictions = self.model(X_batch)
            loss = self.loss_fn(predictions, y_batch)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm
                )

            # Optimization step
            self.optimizer.step()

            total_loss += loss.item()
            batch_count += 1

            if (batch_idx + 1) % 10 == 0:
                avg_loss = total_loss / batch_count
                logger.debug(f"Batch {batch_idx + 1}: Loss = {avg_loss:.6f}")

        epoch_loss = total_loss / batch_count
        return {'train_loss': epoch_loss}

    def validate(self) -> Dict[str, float]:
        """
        Validate model on validation set.

        Returns:
            Dict with validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        batch_count = 0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for X_batch, y_batch in self.val_dataloader:
                X_batch = X_batch.to(self.device).float()
                y_batch = y_batch.to(self.device).float()

                predictions = self.model(X_batch)
                loss = self.loss_fn(predictions, y_batch)

                total_loss += loss.item()
                batch_count += 1

                all_predictions.append(predictions.cpu())
                all_targets.append(y_batch.cpu())

        val_loss = total_loss / batch_count

        # Calculate additional metrics
        predictions_np = torch.cat(all_predictions).numpy().flatten()
        targets_np = torch.cat(all_targets).numpy().flatten()

        mae = np.mean(np.abs(predictions_np - targets_np))
        rmse = np.sqrt(np.mean((predictions_np - targets_np) ** 2))

        return {
            'val_loss': val_loss,
            'val_mae': mae,
            'val_rmse': rmse
        }

    def fit(
        self,
        epochs: int = 100,
        early_stopping_patience: int = 15,
        log_interval: int = 1
    ) -> Dict:
        """
        Train model for specified epochs.

        Args:
            epochs: Number of training epochs
            early_stopping_patience: Patience for early stopping
            log_interval: Epochs between logging

        Returns:
            Dict with training history
        """
        logger.info(f"Starting training for {epochs} epochs")

        if self.early_stopping is None:
            self.set_early_stopping(patience=early_stopping_patience)

        for epoch in range(epochs):
            self.epoch = epoch + 1

            # Training
            train_metrics = self.train_epoch()

            # Validation
            val_metrics = self.validate()

            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['val_loss'])
                else:
                    self.scheduler.step()

            # Record history
            self.training_history['train_loss'].append(train_metrics['train_loss'])
            self.training_history['val_loss'].append(val_metrics['val_loss'])
            self.training_history['learning_rate'].append(
                self.optimizer.param_groups[0]['lr']
            )

            # Checkpointing
            if self.checkpoint is not None:
                self.checkpoint.save(self.model, val_metrics['val_loss'])

            # Logging
            if (self.epoch % log_interval) == 0:
                logger.info(
                    f"Epoch {self.epoch:3d} | "
                    f"Train Loss: {train_metrics['train_loss']:.6f} | "
                    f"Val Loss: {val_metrics['val_loss']:.6f} | "
                    f"MAE: {val_metrics['val_mae']:.6f} | "
                    f"LR: {self.optimizer.param_groups[0]['lr']:.6f}"
                )

            # Early stopping
            if self.early_stopping(val_metrics['val_loss']):
                logger.info(f"Early stopping at epoch {self.epoch}")
                break

        logger.info("Training completed")

        return {
            'final_epoch': self.epoch,
            'best_val_loss': min(self.training_history['val_loss']),
            'history': self.training_history
        }

    def save_checkpoint(self, path: str):
        """
        Save training checkpoint.

        Args:
            path: Path to save checkpoint
        """
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_history': self.training_history,
            'best_val_loss': self.best_val_loss
        }
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """
        Load training checkpoint.

        Args:
            path: Path to checkpoint
        """
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.training_history = checkpoint['training_history']
        self.best_val_loss = checkpoint['best_val_loss']
        logger.info(f"Checkpoint loaded from {path}")

    def get_summary(self) -> Dict:
        """
        Get training summary.

        Returns:
            Dict with training summary
        """
        return {
            'total_epochs': self.epoch,
            'best_val_loss': min(self.training_history['val_loss']),
            'final_train_loss': self.training_history['train_loss'][-1],
            'final_val_loss': self.training_history['val_loss'][-1],
            'model_params': sum(p.numel() for p in self.model.parameters())
        }


def create_optimizer(
    model: nn.Module,
    optimizer_type: str = 'adam',
    learning_rate: float = 0.001,
    **kwargs
) -> optim.Optimizer:
    """
    Create optimizer for training.

    Args:
        model: Model to optimize
        optimizer_type: 'adam', 'sgd', 'rmsprop'
        learning_rate: Learning rate
        **kwargs: Additional optimizer arguments

    Returns:
        Optimizer instance
    """
    params = model.parameters()

    if optimizer_type.lower() == 'adam':
        return optim.Adam(params, lr=learning_rate, **kwargs)
    elif optimizer_type.lower() == 'sgd':
        return optim.SGD(params, lr=learning_rate, **kwargs)
    elif optimizer_type.lower() == 'rmsprop':
        return optim.RMSprop(params, lr=learning_rate, **kwargs)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")


def create_scheduler(
    optimizer: optim.Optimizer,
    scheduler_type: str = 'step',
    **kwargs
):
    """
    Create learning rate scheduler.

    Args:
        optimizer: Optimizer to schedule
        scheduler_type: 'step', 'cosine', 'plateau'
        **kwargs: Additional scheduler arguments

    Returns:
        Scheduler instance
    """
    if scheduler_type.lower() == 'step':
        step_size = kwargs.get('step_size', 10)
        gamma = kwargs.get('gamma', 0.5)
        return StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_type.lower() == 'cosine':
        T_max = kwargs.get('T_max', 100)
        eta_min = kwargs.get('eta_min', 0)
        return CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    elif scheduler_type.lower() == 'plateau':
        mode = kwargs.get('mode', 'min')
        factor = kwargs.get('factor', 0.5)
        patience = kwargs.get('patience', 10)
        return ReduceLROnPlateau(
            optimizer, mode=mode, factor=factor, patience=patience
        )
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_type}")
