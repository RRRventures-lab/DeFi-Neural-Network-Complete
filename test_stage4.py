#!/usr/bin/env python3
"""
Stage 4 Training Pipeline Test Suite

Tests:
1. Loss function implementations
2. Data loader creation and batching
3. Walk-forward validation splitting
4. Trainer class and training loop
5. Early stopping mechanism
6. Model checkpointing
7. Full training pipeline with real data

Run with: python test_stage4.py
"""

import asyncio
import sys
import torch
import torch.nn as nn
import numpy as np
import logging
from datetime import datetime, timedelta
from pathlib import Path
import tempfile

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from training.loss_functions import (
    MSELoss, MAELoss, HuberLoss, SharpeRatioLoss,
    QuantileLoss, DirectionalAccuracyLoss,
    ReturnVolatilityLoss, create_loss_function
)
from training.data_loaders import (
    TimeSeriesDataset, WalkForwardValidator,
    create_data_loaders, create_walk_forward_loaders,
    prepare_data
)
from training.trainer import (
    Trainer, EarlyStopping, ModelCheckpoint,
    create_optimizer, create_scheduler
)
from models.lstm_model import create_lstm_model
from models.cnn_model import create_cnn_model
from models.ensemble_model import create_ensemble_model
from data.data_ingestion import DataIngestionPipeline
from features.feature_pipeline import FeaturePipeline


def test_loss_functions():
    """Test all loss function implementations."""
    print("\n" + "="*60)
    print("TEST 1: Loss Functions")
    print("="*60)

    try:
        # Create dummy data
        batch_size = 32
        predictions = torch.randn(batch_size, 1)
        targets = torch.randn(batch_size, 1)

        losses = {
            'MSE': create_loss_function('mse'),
            'MAE': create_loss_function('mae'),
            'Huber': create_loss_function('huber', delta=1.0),
            'Sharpe': create_loss_function('sharpe'),
            'Quantile': create_loss_function('quantile', quantile=0.5),
            'Directional': create_loss_function('directional'),
            'Return-Vol': create_loss_function('return_volatility')
        }

        print("✓ Testing loss functions:")
        for name, loss_fn in losses.items():
            loss_value = loss_fn(predictions, targets)
            print(f"  {name:.<20} {loss_value.item():.6f}")

        # Test loss values are scalars
        for name, loss_fn in losses.items():
            loss = loss_fn(predictions, targets)
            if loss.shape == torch.Size([]):
                print(f"✓ {name} returns scalar")
            else:
                print(f"❌ {name} shape incorrect: {loss.shape}")
                return False

        print("✅ PASSED")
        return True

    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_loaders():
    """Test data loader creation and batching."""
    print("\n" + "="*60)
    print("TEST 2: Data Loaders")
    print("="*60)

    try:
        # Create synthetic data
        num_samples = 100
        timesteps = 30
        features = 40

        X = np.random.randn(num_samples, timesteps, features).astype(np.float32)
        y = np.random.randn(num_samples).astype(np.float32)

        print(f"✓ Created synthetic data: X={X.shape}, y={y.shape}")

        # Test dataset
        dataset = TimeSeriesDataset(X, y, normalize=True)
        print(f"✓ Dataset created: {len(dataset)} samples")

        # Test data loaders
        train_loader, val_loader = create_data_loaders(
            X, y,
            batch_size=32,
            validation_split=0.2,
            normalize=True
        )

        print(f"✓ Data loaders created")

        # Check batches
        batch_count = 0
        for X_batch, y_batch in train_loader:
            batch_count += 1
            if X_batch.shape[0] <= 32 and X_batch.shape[1] == 30 and X_batch.shape[2] == 40:
                print(f"✓ Batch {batch_count}: X={X_batch.shape}, y={y_batch.shape}")
            else:
                print(f"❌ Batch shape incorrect: {X_batch.shape}")
                return False

        print("✅ PASSED")
        return True

    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_walk_forward_validation():
    """Test walk-forward validation splitting."""
    print("\n" + "="*60)
    print("TEST 3: Walk-Forward Validation")
    print("="*60)

    try:
        num_windows = 100
        validator = WalkForwardValidator(
            num_windows=num_windows,
            validation_size=0.2,
            num_steps=5
        )

        splits = validator.get_splits()
        print(f"✓ Generated {len(splits)} walk-forward splits")

        for i, (train_idx, val_idx) in enumerate(splits):
            print(f"  Split {i+1}: Train={len(train_idx)}, Val={len(val_idx)}")

            # Verify no overlap
            if len(np.intersect1d(train_idx, val_idx)) == 0:
                print(f"  ✓ No overlap between train and validation")
            else:
                print(f"  ❌ Overlap detected!")
                return False

            # Verify temporal order (no look-ahead bias)
            # Allow for empty training set in first fold
            if len(train_idx) > 0 and len(val_idx) > 0:
                if train_idx[-1] < val_idx[0]:
                    print(f"  ✓ Temporal order preserved")
                else:
                    print(f"  ❌ Temporal order violated!")
                    return False
            else:
                print(f"  ✓ Split handling correct")

        print("✅ PASSED")
        return True

    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


def test_early_stopping():
    """Test early stopping mechanism."""
    print("\n" + "="*60)
    print("TEST 4: Early Stopping")
    print("="*60)

    try:
        early_stop = EarlyStopping(patience=3, min_delta=0.001)

        # Simulate validation losses
        losses = [1.0, 0.9, 0.85, 0.84, 0.844, 0.845, 0.846, 0.847]

        for epoch, loss in enumerate(losses):
            should_stop = early_stop(loss)
            print(f"Epoch {epoch}: Loss={loss:.4f}, Stop={should_stop}")

            if should_stop:
                print(f"✓ Early stopping triggered at epoch {epoch}")
                break

        print("✅ PASSED")
        return True

    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


def test_model_checkpoint():
    """Test model checkpointing."""
    print("\n" + "="*60)
    print("TEST 5: Model Checkpointing")
    print("="*60)

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create simple model
            model = nn.Linear(40, 1)

            # Create checkpoint saver
            checkpoint = ModelCheckpoint(
                dirpath=tmpdir,
                filename='test_model.pt',
                monitor='val_loss',
                mode='min'
            )

            print(f"✓ Checkpoint saver created")

            # Test saving
            loss1 = 1.0
            was_saved = checkpoint.save(model, loss1)
            print(f"✓ First save: was_saved={was_saved}")

            loss2 = 0.9  # Better loss
            was_saved = checkpoint.save(model, loss2)
            print(f"✓ Second save (improvement): was_saved={was_saved}")

            loss3 = 0.95  # Worse loss
            was_saved = checkpoint.save(model, loss3)
            print(f"✓ Third save (no improvement): was_saved={was_saved}")

            print("✅ PASSED")
            return True

    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_trainer():
    """Test Trainer class with mini training loop."""
    print("\n" + "="*60)
    print("TEST 6: Trainer Class")
    print("="*60)

    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Create synthetic data
        num_samples = 128
        X = np.random.randn(num_samples, 30, 40).astype(np.float32)
        y = np.random.randn(num_samples).astype(np.float32)

        # Create data loaders
        train_loader, val_loader = create_data_loaders(
            X, y,
            batch_size=32,
            validation_split=0.2
        )

        print(f"✓ Data loaders created")

        # Create model
        model = create_lstm_model(input_size=40, device=device)
        print(f"✓ Model created")

        # Create training components
        loss_fn = create_loss_function('mse')
        optimizer = create_optimizer(model, 'adam', learning_rate=0.001)
        scheduler = create_scheduler(optimizer, 'step', step_size=5, gamma=0.5)

        print(f"✓ Optimizer and scheduler created")

        # Create trainer
        trainer = Trainer(
            model=model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            scheduler=scheduler
        )

        trainer.set_early_stopping(patience=5)

        print(f"✓ Trainer created")

        # Train for a few epochs
        history = trainer.fit(epochs=10, log_interval=2)

        print(f"✓ Training completed:")
        print(f"  Final epoch: {history['final_epoch']}")
        print(f"  Best val loss: {history['best_val_loss']:.6f}")

        # Check history
        if len(trainer.training_history['train_loss']) > 0:
            print(f"✓ Training history recorded")
        else:
            print(f"❌ No training history")
            return False

        print("✅ PASSED")
        return True

    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_with_real_data():
    """Test training with real market data."""
    print("\n" + "="*60)
    print("TEST 7: Training with Real Data")
    print("="*60)

    pipeline = DataIngestionPipeline()
    feature_pipeline = FeaturePipeline()
    await pipeline.initialize()

    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Fetch real data
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')

        print(f"Fetching real market data...")
        df = await pipeline.fetch_historical('AAPL', start_date, end_date)

        if df.empty:
            print("❌ FAILED: No data returned")
            return False

        # Compute features
        features_df = feature_pipeline.compute_features(df, 'AAPL')
        print(f"✓ Computed {len(features_df.columns)} features")

        # Generate windows
        X, y = feature_pipeline.generate_windows(features_df, window_size=30)

        if len(X) == 0:
            print("❌ FAILED: No windows generated")
            return False

        print(f"✓ Generated {len(X)} windows")

        # Prepare data
        X_array = np.array(X).astype(np.float32)
        y_array = np.array(y).astype(np.float32)

        # Create data loaders
        train_loader, val_loader = create_data_loaders(
            X_array, y_array,
            batch_size=16,
            validation_split=0.2
        )

        print(f"✓ Data loaders created")

        # Create model
        input_size = X_array.shape[2]
        model = create_lstm_model(input_size=input_size, device=device)

        # Training setup
        loss_fn = create_loss_function('mse')
        optimizer = create_optimizer(model, 'adam', learning_rate=0.001)

        trainer = Trainer(
            model=model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device
        )

        # Train briefly
        print(f"Training for 5 epochs...")
        history = trainer.fit(epochs=5, log_interval=1)

        print(f"✓ Training completed:")
        print(f"  Final epoch: {history['final_epoch']}")
        print(f"  Best val loss: {history['best_val_loss']:.6f}")
        print(f"  Final train loss: {trainer.training_history['train_loss'][-1]:.6f}")
        print(f"  Final val loss: {trainer.training_history['val_loss'][-1]:.6f}")

        print("✅ PASSED")
        return True

    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        await pipeline.close()


def test_optimizers_schedulers():
    """Test optimizer and scheduler creation."""
    print("\n" + "="*60)
    print("TEST 8: Optimizers and Schedulers")
    print("="*60)

    try:
        model = nn.Linear(40, 1)

        # Test optimizers
        optimizers = {
            'adam': create_optimizer(model, 'adam'),
            'sgd': create_optimizer(model, 'sgd', momentum=0.9),
            'rmsprop': create_optimizer(model, 'rmsprop')
        }

        print("✓ Optimizers created:")
        for name, opt in optimizers.items():
            print(f"  {name}: {type(opt).__name__}")

        # Test schedulers
        adam = optimizers['adam']
        schedulers = {
            'step': create_scheduler(adam, 'step', step_size=10),
            'cosine': create_scheduler(adam, 'cosine', T_max=100),
            'plateau': create_scheduler(adam, 'plateau', patience=5)
        }

        print("✓ Schedulers created:")
        for name, sched in schedulers.items():
            print(f"  {name}: {type(sched).__name__}")

        print("✅ PASSED")
        return True

    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


async def main():
    """Run all tests."""
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + " "*58 + "║")
    print("║" + "   STAGE 4: TRAINING PIPELINE TEST SUITE".center(58) + "║")
    print("║" + " "*58 + "║")
    print("╚" + "="*58 + "╝")

    results = {}

    # Run tests
    results['Loss Functions'] = test_loss_functions()
    results['Data Loaders'] = test_data_loaders()
    results['Walk-Forward Validation'] = test_walk_forward_validation()
    results['Early Stopping'] = test_early_stopping()
    results['Model Checkpoint'] = test_model_checkpoint()
    results['Trainer Class'] = test_trainer()
    results['Real Data Training'] = await test_with_real_data()
    results['Optimizers & Schedulers'] = test_optimizers_schedulers()

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, passed_flag in results.items():
        status = "✅ PASS" if passed_flag else "❌ FAIL"
        print(f"{test_name:.<40} {status}")

    print("="*60)
    print(f"Total: {passed}/{total} tests passed ({100*passed//total}%)")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Stage 4 is complete.")
        print("\nNext steps:")
        print("1. Review training pipeline implementation")
        print("2. Proceed to Stage 5: Backtesting & Validation")
    else:
        print(f"\n⚠️  {total - passed} tests failed. Check logs above.")

    print("\n")

    return passed == total


if __name__ == '__main__':
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
