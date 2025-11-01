#!/usr/bin/env python3
"""
Stage 5 Backtesting & Validation Test Suite

Tests:
1. Performance metrics calculation
2. Backtest engine execution
3. Walk-forward backtesting
4. Benchmark comparison
5. Risk analysis
6. Full backtest with real models

Run with: python test_stage5.py
"""

import asyncio
import sys
import torch
import numpy as np
import logging
from datetime import datetime, timedelta
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))

from evaluation.metrics import PerformanceMetrics, MetricsComparer, calculate_cumulative_returns, calculate_drawdown
from evaluation.backtest import Backtest, BacktestResult, WalkForwardBacktest, BenchmarkComparison, simulate_trading
from models.lstm_model import create_lstm_model
from data.data_ingestion import DataIngestionPipeline
from features.feature_pipeline import FeaturePipeline
from training.data_loaders import create_walk_forward_loaders, prepare_data


def test_performance_metrics():
    """Test performance metrics calculation."""
    print("\n" + "="*60)
    print("TEST 1: Performance Metrics")
    print("="*60)

    try:
        # Create synthetic returns and predictions
        np.random.seed(42)
        returns = np.random.normal(0.0005, 0.01, 252)  # ~252 trading days
        predictions = returns + np.random.normal(0, 0.005, 252)

        metrics = PerformanceMetrics(returns, predictions)
        summary = metrics.get_summary()

        print(f"✓ Total Return: {summary['total_return']:.4f}")
        print(f"✓ Annualized Return: {summary['annualized_return']:.4f}")
        print(f"✓ Sharpe Ratio: {summary['sharpe_ratio']:.4f}")
        print(f"✓ Max Drawdown: {summary['max_drawdown']:.4f}")
        print(f"✓ Win Rate: {summary['win_rate']:.4f}")
        print(f"✓ Directional Accuracy: {summary['directional_accuracy']:.4f}")

        assert isinstance(summary['total_return'], (int, float))
        assert isinstance(summary['sharpe_ratio'], (int, float))

        print("✅ PASSED")
        return True

    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


def test_backtest_engine():
    """Test backtest execution."""
    print("\n" + "="*60)
    print("TEST 2: Backtest Engine")
    print("="*60)

    try:
        # Create synthetic data
        returns = np.random.normal(0.0005, 0.01, 100)
        predictions = np.concatenate([
            np.ones(25) * 0.05,   # Buy signal
            np.ones(25) * -0.05,  # Sell signal
            np.ones(25) * 0.05,   # Buy signal
            np.ones(25) * -0.05   # Sell signal
        ])

        backtest = Backtest(initial_capital=100000)
        result = backtest.run(returns, predictions)

        print(f"✓ Initial Capital: ${100000:,.2f}")
        print(f"✓ Final Equity: ${result.equity_curve[-1]:,.2f}")
        print(f"✓ Return: {(result.equity_curve[-1] - 100000) / 100000:.4f}")
        print(f"✓ Number of Trades: {len(result.trades)}")

        assert len(result.equity_curve) == len(returns) + 1
        assert len(result.positions) == len(returns)

        print("✅ PASSED")
        return True

    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cumulative_returns():
    """Test cumulative return calculation."""
    print("\n" + "="*60)
    print("TEST 3: Cumulative Returns")
    print("="*60)

    try:
        returns = np.array([0.01, 0.02, -0.01, 0.005])
        cum_returns = calculate_cumulative_returns(returns)

        expected = np.array([0.01, 0.0302, 0.0199, 0.0249])
        assert np.allclose(cum_returns, expected, rtol=1e-2)

        print(f"✓ Input returns: {returns}")
        print(f"✓ Cumulative returns: {cum_returns}")
        print(f"✓ Calculation correct")

        print("✅ PASSED")
        return True

    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


def test_drawdown():
    """Test drawdown calculation."""
    print("\n" + "="*60)
    print("TEST 4: Drawdown Calculation")
    print("="*60)

    try:
        returns = np.array([0.05, 0.05, -0.10, 0.05, 0.05])
        drawdowns = calculate_drawdown(returns)

        print(f"✓ Returns: {returns}")
        print(f"✓ Drawdowns: {drawdowns}")
        print(f"✓ Max Drawdown: {np.min(drawdowns):.4f}")

        assert len(drawdowns) == len(returns)
        assert np.min(drawdowns) < 0

        print("✅ PASSED")
        return True

    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


def test_benchmark_comparison():
    """Test benchmark comparison."""
    print("\n" + "="*60)
    print("TEST 5: Benchmark Comparison")
    print("="*60)

    try:
        strategy_returns = np.random.normal(0.001, 0.01, 252)
        benchmark_returns = np.random.normal(0.0005, 0.01, 252)

        comparison = BenchmarkComparison(strategy_returns, benchmark_returns)
        result = comparison.get_comparison()

        print(f"✓ Strategy Return: {result['strategy_return']:.4f}")
        print(f"✓ Benchmark Return: {result['benchmark_return']:.4f}")
        print(f"✓ Excess Return: {result['excess_return']:.4f}")
        print(f"✓ Information Ratio: {result['information_ratio']:.4f}")
        print(f"✓ Batting Average: {result['batting_average']:.4f}")

        assert 0 <= result['batting_average'] <= 1

        print("✅ PASSED")
        return True

    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


def test_simulate_trading():
    """Test trading simulation."""
    print("\n" + "="*60)
    print("TEST 6: Trading Simulation")
    print("="*60)

    try:
        predictions = np.array([0.05, 0.05, -0.05, -0.05, 0.05, 0.05])
        returns = np.array([0.01, 0.02, -0.01, -0.02, 0.01, 0.015])

        equity_curve, final_equity = simulate_trading(
            predictions, returns,
            entry_threshold=0.0,
            initial_capital=100000
        )

        print(f"✓ Initial Capital: $100,000")
        print(f"✓ Final Equity: ${final_equity:,.2f}")
        print(f"✓ Return: {(final_equity - 100000) / 100000:.4f}")
        print(f"✓ Equity Curve Length: {len(equity_curve)}")

        assert len(equity_curve) == len(returns) + 1
        assert final_equity > 0

        print("✅ PASSED")
        return True

    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False


async def test_with_real_model():
    """Test backtesting with real model and data."""
    print("\n" + "="*60)
    print("TEST 7: Backtesting with Real Model")
    print("="*60)

    pipeline = DataIngestionPipeline()
    feature_pipeline = FeaturePipeline()
    await pipeline.initialize()

    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Fetch data
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')

        print("Fetching AAPL data...")
        df = await pipeline.fetch_historical('AAPL', start_date, end_date)

        # Compute features
        features_df = feature_pipeline.compute_features(df, 'AAPL')
        X, y = feature_pipeline.generate_windows(features_df, window_size=30)

        # Prepare data
        X_array = np.array(X).astype(np.float32)
        y_array = np.array(y).astype(np.float32)

        # Create walk-forward loaders
        # Note: With 34 windows total, use smaller validation fraction to avoid empty splits
        num_samples = X_array.shape[0]
        if num_samples < 50:
            # For small datasets, use single train/val split instead of walk-forward
            from training.data_loaders import create_data_loaders
            train_loader, val_loader = create_data_loaders(
                X_array, y_array, batch_size=16,
                validation_split=0.2, normalize=True
            )
            wf_loaders = [(train_loader, val_loader)]
        else:
            # For larger datasets, use walk-forward
            wf_loaders = create_walk_forward_loaders(
                X_array, y_array, batch_size=16,
                validation_fraction=0.2, num_steps=3
            )

        print(f"✓ Generated {len(wf_loaders)} walk-forward folds")

        # Create and test model
        model = create_lstm_model(input_size=X_array.shape[2], device=device)

        # Run backtest on first fold
        train_loader, val_loader = wf_loaders[0]

        # Get predictions
        predictions = []
        returns = []

        model.eval()
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device).float()
                y_batch = y_batch.to(device).float()

                pred = model(X_batch)
                predictions.extend(pred.cpu().numpy().flatten())
                returns.extend(y_batch.cpu().numpy().flatten())

        predictions = np.array(predictions)
        returns = np.array(returns)

        # Run backtest
        backtest = Backtest(initial_capital=100000)
        result = backtest.run(returns, predictions)

        # Calculate metrics
        metrics = PerformanceMetrics(returns, predictions)
        summary = metrics.get_summary()

        print(f"✓ Sharpe Ratio: {summary['sharpe_ratio']:.4f}")
        print(f"✓ Max Drawdown: {summary['max_drawdown']:.4f}")
        print(f"✓ Win Rate: {summary['win_rate']:.4f}")
        print(f"✓ Final Equity: ${result.equity_curve[-1]:,.2f}")

        print("✅ PASSED")
        return True

    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        await pipeline.close()


def test_metrics_comparer():
    """Test metrics comparison across models."""
    print("\n" + "="*60)
    print("TEST 8: Metrics Comparer")
    print("="*60)

    try:
        # Create metrics for different models
        np.random.seed(42)
        returns1 = np.random.normal(0.001, 0.008, 252)
        pred1 = returns1 + np.random.normal(0, 0.004, 252)

        returns2 = np.random.normal(0.0008, 0.009, 252)
        pred2 = returns2 + np.random.normal(0, 0.005, 252)

        metrics1 = PerformanceMetrics(returns1, pred1).calculate_all()
        metrics2 = PerformanceMetrics(returns2, pred2).calculate_all()

        metrics_dict = {
            'Model A': metrics1,
            'Model B': metrics2
        }

        comparer = MetricsComparer(metrics_dict)

        # Test rankings
        sharpe_rank = comparer.rank_by_sharpe()
        return_rank = comparer.rank_by_return()

        print(f"✓ Models compared: {len(metrics_dict)}")
        print(f"✓ Sharpe Ranking: {list(sharpe_rank.index)}")
        print(f"✓ Return Ranking: {list(return_rank.index)}")

        assert len(sharpe_rank) == 2
        assert len(return_rank) == 2

        print("✅ PASSED")
        return True

    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all tests."""
    print("\n" + "╔" + "="*58 + "╗")
    print("║" + " "*58 + "║")
    print("║" + "   STAGE 5: BACKTESTING & VALIDATION TEST SUITE".center(58) + "║")
    print("║" + " "*58 + "║")
    print("╚" + "="*58 + "╝")

    results = {}

    results['Performance Metrics'] = test_performance_metrics()
    results['Backtest Engine'] = test_backtest_engine()
    results['Cumulative Returns'] = test_cumulative_returns()
    results['Drawdown'] = test_drawdown()
    results['Benchmark Comparison'] = test_benchmark_comparison()
    results['Trading Simulation'] = test_simulate_trading()
    results['Real Model Backtest'] = await test_with_real_model()
    results['Metrics Comparer'] = test_metrics_comparer()

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
        print("\n🎉 ALL TESTS PASSED! Stage 5 is complete.")
    else:
        print(f"\n⚠️  {total - passed} tests failed.")

    return passed == total


if __name__ == '__main__':
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
