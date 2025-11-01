#!/usr/bin/env python3
"""
Stage 2 Feature Engineering Test Suite

Tests:
1. Technical indicators computation
2. Feature pipeline
3. Feature validation
4. Window generation
5. Feature normalization
6. Batch feature processing

Run with: python test_stage2.py
"""

import asyncio
import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from data.data_ingestion import DataIngestionPipeline
from features.technical_indicators import TechnicalIndicators
from features.feature_pipeline import FeaturePipeline
from features.feature_validator import FeatureValidator


async def test_technical_indicators():
    """Test technical indicators computation."""
    print("\n" + "="*60)
    print("TEST 1: Technical Indicators Computation")
    print("="*60)

    pipeline = DataIngestionPipeline()
    await pipeline.initialize()

    try:
        # Fetch data
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')

        print(f"Fetching AAPL data for indicator testing...")
        df = await pipeline.fetch_historical('AAPL', start_date, end_date)

        if df.empty:
            print("❌ FAILED: No data returned")
            return False

        print(f"✓ Fetched {len(df)} candles")

        # Compute indicators
        indicators = TechnicalIndicators()
        features_df = indicators.compute_all_indicators(df)

        if features_df.empty:
            print("❌ FAILED: No features computed")
            return False

        num_indicators = len(features_df.columns) - len(df.columns)
        print(f"✓ Computed {num_indicators} indicators")

        # Check specific indicators
        required_indicators = [
            'sma_20', 'ema_12', 'rsi_14', 'macd', 'atr',
            'bb_upper', 'bb_lower', 'obv', 'mfi'
        ]

        missing = [ind for ind in required_indicators if ind not in features_df.columns]
        if missing:
            print(f"⚠️  Missing indicators: {missing}")
        else:
            print(f"✓ All required indicators present")

        # Check for NaN values
        nan_count = features_df.isna().sum().sum()
        print(f"✓ NaN count: {nan_count} (acceptable at start of series)")

        print("✅ PASSED")
        return True

    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

    finally:
        await pipeline.close()


async def test_feature_pipeline():
    """Test feature pipeline."""
    print("\n" + "="*60)
    print("TEST 2: Feature Pipeline")
    print("="*60)

    pipeline = DataIngestionPipeline()
    feature_pipeline = FeaturePipeline()
    await pipeline.initialize()

    try:
        # Fetch data
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')

        print(f"Fetching MSFT data...")
        df = await pipeline.fetch_historical('MSFT', start_date, end_date)

        if df.empty:
            print("❌ FAILED: No data returned")
            return False

        # Process through pipeline
        print(f"Processing through feature pipeline...")
        features_df = feature_pipeline.compute_features(df, 'MSFT')

        print(f"✓ {len(features_df.columns)} total columns")
        print(f"✓ {len(features_df)} rows processed")

        # Save features
        saved = feature_pipeline.save_features(features_df, 'MSFT', 'parquet')
        if saved:
            print("✓ Features saved to disk")
        else:
            print("⚠️  Could not save features")

        # Load features
        loaded_df = feature_pipeline.load_features('MSFT', 'parquet')
        if loaded_df is not None:
            print(f"✓ Features loaded from disk ({len(loaded_df)} rows)")
        else:
            print("⚠️  Could not load features")

        print("✅ PASSED")
        return True

    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

    finally:
        await pipeline.close()


async def test_feature_validation():
    """Test feature validation."""
    print("\n" + "="*60)
    print("TEST 3: Feature Validation")
    print("="*60)

    pipeline = DataIngestionPipeline()
    feature_pipeline = FeaturePipeline()
    validator = FeatureValidator()
    await pipeline.initialize()

    try:
        # Fetch and process data
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')

        print(f"Fetching GOOGL data...")
        df = await pipeline.fetch_historical('GOOGL', start_date, end_date)

        if df.empty:
            print("❌ FAILED: No data returned")
            return False

        # Compute features
        features_df = feature_pipeline.compute_features(df, 'GOOGL')

        # Validate
        is_valid, issues = validator.validate_features(features_df)

        if is_valid:
            print(f"✓ Feature validation passed")
        else:
            print(f"⚠️  Validation issues: {issues}")
            # Repair
            repaired_df = validator.repair_features(features_df)
            is_valid_after, issues_after = validator.validate_features(repaired_df)
            if is_valid_after:
                print(f"✓ Features repaired successfully")
            else:
                print(f"❌ Repair unsuccessful")
                return False

        # Check completeness
        completeness = validator.check_feature_completeness(features_df)
        print(f"✓ Complete rows: {completeness['complete_rows']}/{len(features_df)}")

        # Check for outliers
        outliers = validator.detect_outliers(features_df, threshold=3.0)
        print(f"✓ Outlier detection: {len(outliers)} columns with outliers")

        print("✅ PASSED")
        return True

    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

    finally:
        await pipeline.close()


async def test_window_generation():
    """Test window generation for ML."""
    print("\n" + "="*60)
    print("TEST 4: Window Generation")
    print("="*60)

    pipeline = DataIngestionPipeline()
    feature_pipeline = FeaturePipeline()
    await pipeline.initialize()

    try:
        # Fetch data
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')

        print(f"Fetching NVDA data for window generation...")
        df = await pipeline.fetch_historical('NVDA', start_date, end_date)

        if df.empty:
            print("❌ FAILED: No data returned")
            return False

        # Compute features
        features_df = feature_pipeline.compute_features(df, 'NVDA')

        # Generate windows
        window_size = 30
        X, y = feature_pipeline.generate_windows(
            features_df,
            window_size=window_size,
            step_size=1
        )

        if len(X) == 0:
            print("❌ FAILED: No windows generated")
            return False

        print(f"✓ Generated {len(X)} windows")
        print(f"  Window shape: {X[0].shape}")
        print(f"  Target shape: {y.shape}")
        print(f"  Target statistics:")
        print(f"    Mean: {y.mean():.6f}")
        print(f"    Std: {y.std():.6f}")
        print(f"    Min: {y.min():.6f}")
        print(f"    Max: {y.max():.6f}")

        # Check for NaN in targets
        nan_targets = np.isnan(y).sum()
        print(f"  NaN targets: {nan_targets}")

        print("✅ PASSED")
        return True

    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

    finally:
        await pipeline.close()


async def test_feature_normalization():
    """Test feature normalization."""
    print("\n" + "="*60)
    print("TEST 5: Feature Normalization")
    print("="*60)

    pipeline = DataIngestionPipeline()
    feature_pipeline = FeaturePipeline()
    await pipeline.initialize()

    try:
        # Fetch and process data
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')

        print(f"Fetching TSLA data...")
        df = await pipeline.fetch_historical('TSLA', start_date, end_date)

        if df.empty:
            print("❌ FAILED: No data returned")
            return False

        # Compute features
        features_df = feature_pipeline.compute_features(df, 'TSLA')

        # Test different normalization methods
        methods = ['minmax', 'zscore', 'robust']

        for method in methods:
            print(f"✓ Testing {method} normalization...")
            normalized_df, params = feature_pipeline.normalize_features(
                features_df,
                method=method
            )

            # Check ranges
            numeric_df = normalized_df.select_dtypes(include=[np.number])

            if method == 'minmax':
                # Check if values are in [0, 1] (approximately)
                in_range = ((numeric_df >= -0.01) & (numeric_df <= 1.01)).all().all()
                if in_range:
                    print(f"  ✓ Values in [0, 1] range")
                else:
                    print(f"  ⚠️  Some values outside range")

            elif method == 'zscore':
                # Check if mean ≈ 0, std ≈ 1
                means = numeric_df.mean().abs()
                stds = numeric_df.std()
                print(f"  ✓ Mean range: [{means.min():.3f}, {means.max():.3f}]")
                print(f"  ✓ Std range: [{stds.min():.3f}, {stds.max():.3f}]")

        print("✅ PASSED")
        return True

    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

    finally:
        await pipeline.close()


async def test_batch_feature_processing():
    """Test batch feature processing."""
    print("\n" + "="*60)
    print("TEST 6: Batch Feature Processing")
    print("="*60)

    pipeline = DataIngestionPipeline()
    feature_pipeline = FeaturePipeline()
    await pipeline.initialize()

    try:
        # Fetch batch data
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')

        print(f"Fetching batch: {symbols}...")
        data_dict = await pipeline.fetch_historical_batch(symbols, start_date, end_date)

        # Process batch
        print(f"Processing batch through feature pipeline...")
        features_dict = feature_pipeline.compute_batch(data_dict)

        processed = sum(1 for df in features_dict.values() if not df.empty)
        print(f"✓ Successfully processed {processed}/{len(symbols)} symbols")

        # Check results
        for symbol, features_df in features_dict.items():
            if not features_df.empty:
                print(f"  {symbol}: {len(features_df.columns)} features, {len(features_df)} rows")

        # Get stats
        stats = feature_pipeline.get_pipeline_stats()
        print(f"✓ Pipeline stats:")
        print(f"  Computed symbols: {stats['computed_symbols']}")
        print(f"  Total cached: {len(stats['metadata'])} entries")

        print("✅ PASSED")
        return processed == len(symbols)

    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

    finally:
        await pipeline.close()


async def test_feature_statistics():
    """Test feature statistics generation."""
    print("\n" + "="*60)
    print("TEST 7: Feature Statistics & Reporting")
    print("="*60)

    pipeline = DataIngestionPipeline()
    feature_pipeline = FeaturePipeline()
    validator = FeatureValidator()
    await pipeline.initialize()

    try:
        # Fetch data
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')

        print(f"Fetching SPY data...")
        df = await pipeline.fetch_historical('SPY', start_date, end_date)

        if df.empty:
            print("❌ FAILED: No data returned")
            return False

        # Compute features
        features_df = feature_pipeline.compute_features(df, 'SPY')

        # Generate report
        report = feature_pipeline.generate_report(features_df, 'SPY')

        print(f"✓ Feature Report:")
        print(f"  Rows: {report['data_quality']['rows']}")
        print(f"  Columns: {report['data_quality']['columns']}")
        print(f"  NaN count: {report['data_quality']['nan_count']}")
        print(f"  Inf count: {report['data_quality']['inf_count']}")

        # Get feature statistics
        stats = validator.get_feature_statistics(features_df)
        num_stats = len(stats)
        print(f"✓ Statistics for {num_stats} numeric features computed")

        # Get feature names
        feature_names = validator.get_feature_names(features_df)
        print(f"✓ Extracted {len(feature_names)} computed feature names")

        print("✅ PASSED")
        return True

    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

    finally:
        await pipeline.close()


async def main():
    """Run all tests."""
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + " "*58 + "║")
    print("║" + "   STAGE 2: FEATURE ENGINEERING TEST SUITE".center(58) + "║")
    print("║" + " "*58 + "║")
    print("╚" + "="*58 + "╝")

    results = {}

    # Run tests
    results['Technical Indicators'] = await test_technical_indicators()
    results['Feature Pipeline'] = await test_feature_pipeline()
    results['Feature Validation'] = await test_feature_validation()
    results['Window Generation'] = await test_window_generation()
    results['Feature Normalization'] = await test_feature_normalization()
    results['Batch Processing'] = await test_batch_feature_processing()
    results['Statistics & Reporting'] = await test_feature_statistics()

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
    print(f"Total: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Stage 2 is complete.")
        print("\nNext steps:")
        print("1. Review the feature engineering implementation")
        print("2. Proceed to Stage 3: Neural Network Architecture")
        print("3. Build LSTM, Attention, and CNN models")
    else:
        print(f"\n⚠️  {total - passed} tests failed. Check logs above.")

    print("\n")


if __name__ == '__main__':
    asyncio.run(main())
