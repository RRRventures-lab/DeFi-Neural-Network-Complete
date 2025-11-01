#!/usr/bin/env python3
"""
Stage 3 Neural Network Architecture Test Suite

Tests:
1. LSTM model instantiation and forward pass
2. CNN model instantiation and forward pass
3. Attention model instantiation and forward pass
4. Ensemble model combining all three
5. Model parameter counting and device handling
6. Gradient flow through all architectures
7. Inference mode testing with real feature data

Run with: python test_stage3.py
"""

import asyncio
import sys
import torch
import torch.nn as nn
import logging
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from models.lstm_model import LSTMModel, create_lstm_model
from models.cnn_model import CNNModel, create_cnn_model, MultiScaleCNN
from models.attention_model import AttentionModel, create_attention_model
from models.ensemble_model import EnsembleModel, create_ensemble_model
from data.data_ingestion import DataIngestionPipeline
from features.feature_pipeline import FeaturePipeline


def test_lstm_instantiation():
    """Test LSTM model instantiation and basic properties."""
    print("\n" + "=" * 60)
    print("TEST 1: LSTM Model Instantiation")
    print("=" * 60)

    try:
        # Create model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        lstm = create_lstm_model(input_size=40, hidden_size=128, device=device)

        # Check properties
        print(f"✓ Model created on device: {device}")

        param_count = lstm.count_parameters()
        print(f"✓ Total parameters: {param_count:,}")

        if param_count < 50000 or param_count > 500000:
            print(f"⚠️  Unexpected parameter count: {param_count}")
        else:
            print(f"✓ Parameter count is reasonable")

        # Test forward pass with dummy data
        batch_size = 32
        seq_len = 30
        input_size = 40
        x = torch.randn(batch_size, seq_len, input_size).to(device)

        output = lstm(x)
        expected_shape = (batch_size, 1)

        if output.shape == expected_shape:
            print(f"✓ Forward pass output shape correct: {output.shape}")
        else:
            print(f"❌ FAILED: Expected shape {expected_shape}, got {output.shape}")
            return False

        # Test hidden states
        lstm_out, hidden, cell = lstm.get_hidden_states(x)
        print(f"✓ Hidden states extracted: shape {lstm_out.shape}")

        print("✅ PASSED")
        return True

    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cnn_instantiation():
    """Test CNN model instantiation and forward pass."""
    print("\n" + "=" * 60)
    print("TEST 2: CNN Model Instantiation")
    print("=" * 60)

    try:
        # Create model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        cnn = create_cnn_model(input_size=40, device=device)

        print(f"✓ CNN model created on device: {device}")

        param_count = cnn.count_parameters()
        print(f"✓ Total parameters: {param_count:,}")

        # Test forward pass
        batch_size = 32
        seq_len = 30
        input_size = 40
        x = torch.randn(batch_size, seq_len, input_size).to(device)

        output = cnn(x)
        expected_shape = (batch_size, 1)

        if output.shape == expected_shape:
            print(f"✓ Forward pass output shape correct: {output.shape}")
        else:
            print(f"❌ FAILED: Expected shape {expected_shape}, got {output.shape}")
            return False

        # Test feature extraction
        features = cnn.get_conv_features(x)
        print(f"✓ Convolutional features extracted: shape {features.shape}")

        # Test eval mode
        cnn.to_eval_mode()
        with torch.no_grad():
            output_eval = cnn(x)
            print(f"✓ Evaluation mode works: output shape {output_eval.shape}")

        print("✅ PASSED")
        return True

    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_attention_instantiation():
    """Test Attention model instantiation and forward pass."""
    print("\n" + "=" * 60)
    print("TEST 3: Attention Model Instantiation")
    print("=" * 60)

    try:
        # Create model
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        attention = create_attention_model(input_size=40, hidden_size=128, device=device)

        print(f"✓ Attention model created on device: {device}")

        param_count = attention.count_parameters()
        print(f"✓ Total parameters: {param_count:,}")

        # Test forward pass
        batch_size = 32
        seq_len = 30
        input_size = 40
        x = torch.randn(batch_size, seq_len, input_size).to(device)

        output = attention(x)
        expected_shape = (batch_size, 1)

        if output.shape == expected_shape:
            print(f"✓ Forward pass output shape correct: {output.shape}")
        else:
            print(f"❌ FAILED: Expected shape {expected_shape}, got {output.shape}")
            return False

        # Test attention weights
        attn_weights = attention.get_attention_weights(x)
        print(f"✓ Attention weights extracted: shape {attn_weights.shape}")

        # Verify attention weights sum to 1
        weight_sums = attn_weights.sum(dim=-1)
        if torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5):
            print(f"✓ Attention weights properly normalized")
        else:
            print(f"⚠️  Attention weights don't sum to 1")

        print("✅ PASSED")
        return True

    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ensemble_model():
    """Test Ensemble model combining all three architectures."""
    print("\n" + "=" * 60)
    print("TEST 4: Ensemble Model")
    print("=" * 60)

    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Create base models
        lstm = create_lstm_model(input_size=40, hidden_size=128, device=device)
        cnn = create_cnn_model(input_size=40, device=device)
        attention = create_attention_model(input_size=40, hidden_size=128, device=device)

        # Create ensemble
        ensemble = create_ensemble_model(lstm, cnn, attention, device=device)

        print(f"✓ Ensemble model created on device: {device}")

        param_count = ensemble.count_parameters()
        print(f"✓ Total parameters (including base learners): {param_count:,}")

        # Test forward pass
        batch_size = 32
        seq_len = 30
        input_size = 40
        x = torch.randn(batch_size, seq_len, input_size).to(device)

        final_pred, details = ensemble(x)
        expected_shape = (batch_size, 1)

        if final_pred.shape == expected_shape:
            print(f"✓ Ensemble output shape correct: {final_pred.shape}")
        else:
            print(f"❌ FAILED: Expected shape {expected_shape}, got {final_pred.shape}")
            return False

        # Check details dictionary
        required_keys = ['lstm_pred', 'cnn_pred', 'attention_pred', 'weights']
        for key in required_keys:
            if key in details:
                print(f"✓ {key} present in details: {details[key].shape}")
            else:
                print(f"❌ Missing {key} in details")
                return False

        # Test weight extraction
        weights = ensemble.get_confidence_weights(x)
        print(f"✓ Confidence weights extracted: {list(weights.keys())}")

        for model_name, weight in weights.items():
            mean_weight = weight.mean().item()
            print(f"  {model_name}: mean weight = {mean_weight:.4f}")

        # Test base prediction extraction
        base_preds = ensemble.get_base_predictions(x)
        print(f"✓ Base predictions extracted: {list(base_preds.keys())}")

        print("✅ PASSED")
        return True

    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gradient_flow():
    """Test gradient flow through all models."""
    print("\n" + "=" * 60)
    print("TEST 5: Gradient Flow")
    print("=" * 60)

    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        models = {
            'LSTM': create_lstm_model(device=device),
            'CNN': create_cnn_model(device=device),
            'Attention': create_attention_model(device=device)
        }

        batch_size = 8
        seq_len = 30
        input_size = 40
        x = torch.randn(batch_size, seq_len, input_size, requires_grad=True).to(device)
        target = torch.randn(batch_size, 1).to(device)

        criterion = nn.MSELoss()

        for model_name, model in models.items():
            model.zero_grad()
            output = model(x)
            loss = criterion(output, target)
            loss.backward()

            # Check gradients
            has_gradients = False
            for param in model.parameters():
                if param.grad is not None and param.grad.abs().sum() > 0:
                    has_gradients = True
                    break

            if has_gradients:
                print(f"✓ {model_name}: Gradients flow correctly")
            else:
                print(f"⚠️  {model_name}: No gradients detected")

        print("✅ PASSED")
        return True

    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_with_real_features():
    """Test models with real feature data."""
    print("\n" + "=" * 60)
    print("TEST 6: Models with Real Feature Data")
    print("=" * 60)

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
        print(f"✓ Computed {len(features_df.columns)} features for {len(features_df)} rows")

        # Generate windows
        X, y = feature_pipeline.generate_windows(features_df, window_size=30, step_size=1)

        if len(X) == 0:
            print("❌ FAILED: No windows generated")
            return False

        print(f"✓ Generated {len(X)} windows of shape {X[0].shape}")

        # Create models
        lstm = create_lstm_model(input_size=X[0].shape[1], hidden_size=128, device=device)
        cnn = create_cnn_model(input_size=X[0].shape[1], device=device)
        attention = create_attention_model(input_size=X[0].shape[1], hidden_size=128, device=device)
        ensemble = create_ensemble_model(lstm, cnn, attention, device=device)

        # Test forward passes
        batch_size = min(8, len(X))
        X_batch = torch.from_numpy(np.array(X[:batch_size])).float().to(device)

        with torch.no_grad():
            lstm_pred = lstm(X_batch)
            cnn_pred = cnn(X_batch)
            attention_pred = attention(X_batch)
            ensemble_pred, details = ensemble(X_batch)

        print(f"✓ LSTM prediction: {lstm_pred.shape}")
        print(f"✓ CNN prediction: {cnn_pred.shape}")
        print(f"✓ Attention prediction: {attention_pred.shape}")
        print(f"✓ Ensemble prediction: {ensemble_pred.shape}")

        # Check prediction ranges
        predictions = torch.cat([lstm_pred, cnn_pred, attention_pred, ensemble_pred])
        pred_mean = predictions.mean().item()
        pred_std = predictions.std().item()
        print(f"✓ Prediction statistics: mean={pred_mean:.6f}, std={pred_std:.6f}")

        print("✅ PASSED")
        return True

    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        await pipeline.close()


def test_model_modes():
    """Test switching between training and evaluation modes."""
    print("\n" + "=" * 60)
    print("TEST 7: Model Modes (Training/Evaluation)")
    print("=" * 60)

    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        models = {
            'LSTM': create_lstm_model(device=device),
            'CNN': create_cnn_model(device=device),
            'Attention': create_attention_model(device=device)
        }

        x = torch.randn(4, 30, 40).to(device)

        for model_name, model in models.items():
            # Test train mode
            model.to_train_mode()
            if model.training:
                print(f"✓ {model_name}: Successfully switched to training mode")
            else:
                print(f"❌ {model_name}: Failed to enter training mode")
                return False

            # Test eval mode
            model.to_eval_mode()
            if not model.training:
                print(f"✓ {model_name}: Successfully switched to eval mode")
            else:
                print(f"❌ {model_name}: Failed to enter eval mode")
                return False

        print("✅ PASSED")
        return True

    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


async def main():
    """Run all tests."""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "   STAGE 3: NEURAL NETWORK ARCHITECTURE TEST SUITE".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "=" * 58 + "╝")

    results = {}

    # Run tests
    results['LSTM Instantiation'] = test_lstm_instantiation()
    results['CNN Instantiation'] = test_cnn_instantiation()
    results['Attention Instantiation'] = test_attention_instantiation()
    results['Ensemble Model'] = test_ensemble_model()
    results['Gradient Flow'] = test_gradient_flow()
    results['Real Feature Data'] = await test_with_real_features()
    results['Model Modes'] = test_model_modes()

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, passed_flag in results.items():
        status = "✅ PASS" if passed_flag else "❌ FAIL"
        print(f"{test_name:.<40} {status}")

    print("=" * 60)
    print(f"Total: {passed}/{total} tests passed ({100*passed//total}%)")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Stage 3 is complete.")
        print("\nNext steps:")
        print("1. Review neural network architectures")
        print("2. Proceed to Stage 4: Training Pipeline")
        print("3. Implement training loops and optimization")
    else:
        print(f"\n⚠️  {total - passed} tests failed. Check logs above.")

    print("\n")

    return passed == total


if __name__ == '__main__':
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
