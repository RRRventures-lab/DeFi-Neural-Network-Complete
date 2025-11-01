"""
Data Ingestion Pipeline

Orchestrates data fetching from multiple sources with caching and validation.
"""

import pandas as pd
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
import json

from data.polygon_client import PolygonClient
from data.data_validator import DataValidator
from config.api_config import config
from config.constants import SUPPORTED_SYMBOLS

logger = logging.getLogger(__name__)


class DataIngestionPipeline:
    """
    Manages data ingestion from APIs with caching and validation.

    Features:
    - Automatic data fetching from Polygon.io
    - Local caching to reduce API calls
    - Data validation before storage
    - Batch operations for efficiency
    - Error handling and retry logic
    """

    def __init__(self, cache_dir: str = './data/cache'):
        self.polygon = PolygonClient()
        self.validator = DataValidator()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.cache_dir / 'metadata.json'
        self._load_metadata()

    def _load_metadata(self):
        """Load cache metadata."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}

    def _save_metadata(self):
        """Save cache metadata."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def _get_cache_path(self, symbol: str, start_date: str, end_date: str) -> Path:
        """Get cache file path for symbol and date range."""
        filename = f'{symbol}_{start_date}_{end_date}.parquet'
        return self.cache_dir / filename

    def _is_cache_valid(self, symbol: str, start_date: str, end_date: str) -> bool:
        """Check if cached data is still valid."""
        cache_key = f'{symbol}_{start_date}_{end_date}'

        if cache_key not in self.metadata:
            return False

        cache_time = datetime.fromisoformat(self.metadata[cache_key]['cached_at'])
        age_hours = (datetime.now() - cache_time).total_seconds() / 3600

        # Cache valid for 24 hours, or immediately if end_date is in past
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        if end_dt < datetime.now():
            return age_hours < 24  # Historical data cached for 24 hours
        else:
            return age_hours < 1   # Recent data cached for 1 hour

    async def fetch_historical(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        use_cache: bool = True,
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data for a symbol.

        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            use_cache: Use cached data if available
            force_refresh: Force refresh from API

        Returns:
            DataFrame with OHLCV data
        """
        cache_path = self._get_cache_path(symbol, start_date, end_date)

        # Try cache first
        if use_cache and not force_refresh and self._is_cache_valid(symbol, start_date, end_date):
            try:
                df = pd.read_parquet(cache_path)
                logger.info(f"Loaded from cache: {symbol} ({len(df)} rows)")
                return df
            except Exception as e:
                logger.warning(f"Cache read failed: {e}")

        # Fetch from API
        logger.info(f"Fetching from API: {symbol} ({start_date} to {end_date})")
        df = await self.polygon.get_historical(symbol, start_date, end_date)

        if df.empty:
            logger.error(f"No data returned for {symbol}")
            return df

        # Validate
        valid, issues = self.validator.validate_ohlcv(df)
        if not valid:
            logger.warning(f"Data validation issues for {symbol}: {issues}")
            df = self.validator.repair_ohlcv(df)

        # Cache
        try:
            df.to_parquet(cache_path)
            self.metadata[f'{symbol}_{start_date}_{end_date}'] = {
                'cached_at': datetime.now().isoformat(),
                'rows': len(df),
                'date_range': [start_date, end_date]
            }
            self._save_metadata()
            logger.info(f"Cached: {symbol} ({len(df)} rows)")
        except Exception as e:
            logger.warning(f"Cache write failed: {e}")

        return df

    async def fetch_historical_batch(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        use_cache: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data for multiple symbols concurrently.

        Args:
            symbols: List of symbols
            start_date: Start date
            end_date: End date
            use_cache: Use cached data

        Returns:
            Dict mapping symbol to DataFrame
        """
        logger.info(f"Fetching batch: {len(symbols)} symbols")

        tasks = [
            self.fetch_historical(sym, start_date, end_date, use_cache)
            for sym in symbols
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        data_dict = {}
        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                logger.error(f"Error fetching {symbol}: {result}")
                data_dict[symbol] = pd.DataFrame()
            else:
                data_dict[symbol] = result

        return data_dict

    async def fetch_quotes(self, symbols: List[str]) -> Dict[str, Dict]:
        """
        Fetch latest quotes for multiple symbols.

        Args:
            symbols: List of symbols

        Returns:
            Dict mapping symbol to quote data
        """
        logger.info(f"Fetching quotes: {len(symbols)} symbols")
        quotes = await self.polygon.get_quotes_batch(symbols)
        return quotes

    async def fetch_data_range(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        chunk_months: int = 12,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Fetch data in chunks to handle API limits.

        Args:
            symbol: Stock symbol
            start_date: Start date
            end_date: End date
            chunk_months: Months per chunk
            use_cache: Use cached data

        Returns:
            Combined DataFrame
        """
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')

        all_data = []
        current = start

        logger.info(f"Fetching in {chunk_months}-month chunks: {symbol}")

        while current < end:
            # Calculate chunk end
            chunk_end = current + timedelta(days=30 * chunk_months)
            if chunk_end > end:
                chunk_end = end

            chunk_start_str = current.strftime('%Y-%m-%d')
            chunk_end_str = chunk_end.strftime('%Y-%m-%d')

            df = await self.fetch_historical(
                symbol,
                chunk_start_str,
                chunk_end_str,
                use_cache=use_cache
            )

            if not df.empty:
                all_data.append(df)

            current = chunk_end + timedelta(days=1)

        if all_data:
            return pd.concat(all_data).drop_duplicates().sort_index()
        else:
            return pd.DataFrame()

    async def initialize(self):
        """Initialize async client."""
        await self.polygon.initialize()
        logger.info("Data ingestion pipeline initialized")

    async def close(self):
        """Close async client."""
        await self.polygon.close()
        logger.info("Data ingestion pipeline closed")

    async def test(self) -> Tuple[bool, str]:
        """Test pipeline connectivity."""
        try:
            success, message = await self.polygon.test_connection()
            logger.info(f"Pipeline test: {message}")
            return success, message
        except Exception as e:
            return False, f"Pipeline test failed: {e}"

    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        return {
            'cached_entries': len(self.metadata),
            'cache_dir': str(self.cache_dir),
            'entries': self.metadata
        }

    def clear_cache(self, symbol: Optional[str] = None):
        """
        Clear cached data.

        Args:
            symbol: Clear only this symbol, or None for all
        """
        if symbol:
            # Remove specific symbol
            keys_to_remove = [k for k in self.metadata.keys() if k.startswith(symbol)]
            for key in keys_to_remove:
                cache_path = self.cache_dir / f'{key}.parquet'
                if cache_path.exists():
                    cache_path.unlink()
                del self.metadata[key]
            logger.info(f"Cleared cache for {symbol}")
        else:
            # Clear all
            for cache_file in self.cache_dir.glob('*.parquet'):
                cache_file.unlink()
            self.metadata = {}
            logger.info("Cleared all cache")

        self._save_metadata()


# Singleton instance
_pipeline_instance = None


async def get_pipeline() -> DataIngestionPipeline:
    """Get or create global pipeline instance."""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = DataIngestionPipeline()
        await _pipeline_instance.initialize()
    return _pipeline_instance


async def close_pipeline():
    """Close global pipeline instance."""
    global _pipeline_instance
    if _pipeline_instance:
        await _pipeline_instance.close()
        _pipeline_instance = None
