#!/usr/bin/env python3
"""
Stage 1 Integration Test Script

Tests:
1. API connectivity (Polygon.io)
2. Data fetching
3. Data validation
4. Caching

Run with: python test_stage1.py
"""

import asyncio
import sys
import logging
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

from data.polygon_client import PolygonClient
from data.data_ingestion import DataIngestionPipeline
from data.data_validator import DataValidator
from config.constants import SUPPORTED_SYMBOLS


async def test_polygon_client():
    """Test Polygon.io client connectivity."""
    print("\n" + "="*60)
    print("TEST 1: Polygon.io API Connectivity")
    print("="*60)

    client = PolygonClient()
    await client.initialize()

    try:
        success, message = await client.test_connection()
        print(f"Result: {message}")
        if success:
            print("✅ PASSED")
        else:
            print("❌ FAILED")
        return success
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False
    finally:
        await client.close()


async def test_data_fetching():
    """Test fetching historical data."""
    print("\n" + "="*60)
    print("TEST 2: Fetch Historical Data")
    print("="*60)

    client = PolygonClient()
    await client.initialize()

    try:
        # Fetch last 30 days of AAPL
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')

        print(f"Fetching AAPL data from {start_date} to {end_date}...")
        df = await client.get_historical('AAPL', start_date, end_date)

        if df.empty:
            print("❌ FAILED: No data returned")
            return False

        print(f"✓ Fetched {len(df)} candles")
        print(f"  Date range: {df.index.min()} to {df.index.max()}")
        print(f"  Columns: {', '.join(df.columns)}")
        print(f"  Latest close: ${df['close'].iloc[-1]:.2f}")
        print("✅ PASSED")
        return True

    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False
    finally:
        await client.close()


async def test_data_validation():
    """Test data validation."""
    print("\n" + "="*60)
    print("TEST 3: Data Validation")
    print("="*60)

    client = PolygonClient()
    validator = DataValidator()
    await client.initialize()

    try:
        # Fetch data
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')

        print(f"Fetching MSFT data for validation...")
        df = await client.get_historical('MSFT', start_date, end_date)

        if df.empty:
            print("❌ FAILED: No data returned")
            return False

        # Validate
        is_valid, issues = validator.validate_ohlcv(df)

        if is_valid:
            print(f"✓ Validation passed for {len(df)} candles")
            print("✅ PASSED")
            return True
        else:
            print(f"⚠️  Validation issues detected: {issues}")
            # Try repair
            df_repaired = validator.repair_ohlcv(df)
            is_valid_after, issues_after = validator.validate_ohlcv(df_repaired)
            if is_valid_after:
                print("✓ Data repaired successfully")
                print("✅ PASSED")
                return True
            else:
                print(f"❌ FAILED: Repair unsuccessful - {issues_after}")
                return False

    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False
    finally:
        await client.close()


async def test_data_ingestion():
    """Test data ingestion pipeline."""
    print("\n" + "="*60)
    print("TEST 4: Data Ingestion Pipeline")
    print("="*60)

    pipeline = DataIngestionPipeline()
    await pipeline.initialize()

    try:
        # Test single symbol
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')

        print(f"Fetching GOOGL data via pipeline...")
        df = await pipeline.fetch_historical('GOOGL', start_date, end_date, use_cache=True)

        if df.empty:
            print("❌ FAILED: No data returned")
            return False

        print(f"✓ Fetched {len(df)} candles")
        print(f"  Date range: {df.index.min()} to {df.index.max()}")

        # Test cache
        print(f"Testing cache (should be instant)...")
        import time
        start_time = time.time()
        df2 = await pipeline.fetch_historical('GOOGL', start_date, end_date, use_cache=True)
        cache_time = time.time() - start_time

        if df.equals(df2):
            print(f"✓ Cache retrieval successful ({cache_time*1000:.1f}ms)")
            print("✅ PASSED")
            return True
        else:
            print("❌ FAILED: Cached data mismatch")
            return False

    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False
    finally:
        await pipeline.close()


async def test_batch_operations():
    """Test batch operations."""
    print("\n" + "="*60)
    print("TEST 5: Batch Operations")
    print("="*60)

    pipeline = DataIngestionPipeline()
    await pipeline.initialize()

    try:
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')

        print(f"Fetching batch: {symbols}...")
        data_dict = await pipeline.fetch_historical_batch(symbols, start_date, end_date)

        successful = sum(1 for df in data_dict.values() if not df.empty)
        print(f"✓ Successfully fetched {successful}/{len(symbols)} symbols")

        for symbol, df in data_dict.items():
            if not df.empty:
                print(f"  {symbol}: {len(df)} candles")
            else:
                print(f"  {symbol}: No data")

        if successful == len(symbols):
            print("✅ PASSED")
            return True
        else:
            print("⚠️  Some symbols failed")
            return successful > 0

    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False
    finally:
        await pipeline.close()


async def test_data_quality_report():
    """Test quality report generation."""
    print("\n" + "="*60)
    print("TEST 6: Data Quality Report")
    print("="*60)

    pipeline = DataIngestionPipeline()
    validator = DataValidator()
    await pipeline.initialize()

    try:
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')

        print(f"Fetching NVDA data and generating report...")
        df = await pipeline.fetch_historical('NVDA', start_date, end_date)

        if df.empty:
            print("❌ FAILED: No data returned")
            return False

        report = validator.generate_quality_report(df, 'NVDA')

        print(f"✓ Quality Report:")
        print(f"  Valid: {report['validation']['is_valid']}")
        print(f"  Fresh: {report['freshness']['is_fresh']} (age: {report['freshness']['age_days']} days)")
        print(f"  Sufficient: {report['sufficiency']['is_sufficient']} ({report['sufficiency']['rows']} rows)")
        print(f"  Price Range: ${report['statistics']['price_range']['min']:.2f} - ${report['statistics']['price_range']['max']:.2f}")

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
    print("║" + "   STAGE 1: API INTEGRATION & DATA PIPELINE TESTS".center(58) + "║")
    print("║" + " "*58 + "║")
    print("╚" + "="*58 + "╝")

    results = {}

    # Run tests
    results['API Connectivity'] = await test_polygon_client()
    results['Data Fetching'] = await test_data_fetching()
    results['Data Validation'] = await test_data_validation()
    results['Data Ingestion'] = await test_data_ingestion()
    results['Batch Operations'] = await test_batch_operations()
    results['Quality Report'] = await test_data_quality_report()

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
        print("\n🎉 ALL TESTS PASSED! Stage 1 is complete.")
        print("\nNext steps:")
        print("1. Review the data pipeline implementation")
        print("2. Proceed to Stage 2: Feature Engineering")
        print("3. Create technical indicators module")
    else:
        print(f"\n⚠️  {total - passed} tests failed. Check logs above.")

    print("\n")


if __name__ == '__main__':
    asyncio.run(main())
