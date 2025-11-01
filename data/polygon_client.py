"""
Polygon.io API Client Wrapper

Handles connection to Polygon.io for stock, ETF, and crypto data.
"""

import asyncio
import aiohttp
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import time
from config.api_config import config
import logging

logger = logging.getLogger(__name__)


class PolygonClient:
    """
    Polygon.io API client for fetching financial data.

    Supports:
    - Stock quotes (real-time)
    - Historical OHLCV data
    - Technical indicators
    - Cryptocurrency data
    - Batch operations
    """

    def __init__(self, api_key: str = config.POLYGON_API_KEY):
        self.api_key = api_key
        self.base_url = config.POLYGON_API_BASE
        self.session = None
        self.rate_limit_per_minute = config.RATE_LIMIT_PER_MINUTE
        self.rate_limit_per_second = config.RATE_LIMIT_PER_SECOND
        self.request_times = []
        self._validate_api_key()

    def _validate_api_key(self):
        """Validate that API key is configured."""
        if not self.api_key or self.api_key == '':
            raise ValueError("POLYGON_API_KEY not configured in .env file")
        logger.info(f"Polygon API key loaded: {self.api_key[:10]}...")

    async def initialize(self):
        """Initialize async HTTP session."""
        self.session = aiohttp.ClientSession()
        logger.info("Polygon client initialized")

    async def close(self):
        """Close async HTTP session."""
        if self.session:
            await self.session.close()
            logger.info("Polygon client closed")

    def _check_rate_limit(self):
        """Check and enforce rate limiting."""
        now = time.time()
        # Remove requests older than 60 seconds
        self.request_times = [t for t in self.request_times if now - t < 60]

        if len(self.request_times) >= self.rate_limit_per_minute:
            sleep_time = 60 - (now - self.request_times[0])
            if sleep_time > 0:
                logger.warning(f"Rate limit approaching, sleeping {sleep_time:.1f}s")
                time.sleep(sleep_time)
                self.request_times = []

        self.request_times.append(now)

    async def get_quote(self, symbol: str) -> Dict:
        """
        Get latest quote for a symbol.

        Args:
            symbol: Stock/crypto symbol (e.g., 'AAPL', 'BTC')

        Returns:
            Dict with quote data (price, bid, ask, volume, timestamp)
        """
        if not self.session:
            raise RuntimeError("Client not initialized. Call initialize() first.")

        self._check_rate_limit()

        # Determine if crypto or stock
        if symbol.startswith('X') or symbol in ['BTC', 'ETH', 'SOL']:
            url = f'{self.base_url}/v1/last/crypto/{symbol}/usd'
        else:
            url = f'{self.base_url}/v3/quotes/latest'

        params = {'apiKey': self.api_key}
        if not symbol.startswith('X') and symbol not in ['BTC', 'ETH', 'SOL']:
            params['ticker'] = symbol

        try:
            async with self.session.get(url, params=params, timeout=30) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    logger.debug(f"Quote fetched: {symbol}")
                    return self._parse_quote(data, symbol)
                else:
                    logger.error(f"Quote fetch failed: {symbol} - Status {resp.status}")
                    return {}
        except Exception as e:
            logger.error(f"Error fetching quote for {symbol}: {e}")
            return {}

    def _parse_quote(self, data: Dict, symbol: str) -> Dict:
        """Parse quote response from Polygon API."""
        try:
            if 'results' in data and len(data['results']) > 0:
                result = data['results'][0]
                return {
                    'symbol': symbol,
                    'price': result.get('c') or result.get('price'),
                    'bid': result.get('bid'),
                    'ask': result.get('ask'),
                    'volume': result.get('v'),
                    'timestamp': result.get('t') or result.get('timestamp'),
                    'raw': result
                }
            elif 'last' in data:
                return {
                    'symbol': symbol,
                    'price': data['last'].get('price'),
                    'timestamp': data['last'].get('exchange'),
                    'raw': data['last']
                }
            elif 'price' in data:
                # Handle simple response with just price
                return {
                    'symbol': symbol,
                    'price': data.get('price'),
                    'timestamp': data.get('timestamp'),
                    'raw': data
                }
        except Exception as e:
            logger.error(f"Error parsing quote: {e}")

        return {}

    async def get_historical(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        timespan: str = 'day',
        adjusted: bool = True
    ) -> pd.DataFrame:
        """
        Get historical OHLCV data.

        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            timespan: 'minute', 'hour', 'day', 'week', 'month', 'quarter', 'year'
            adjusted: Use split/dividend adjusted prices

        Returns:
            DataFrame with columns: open, high, low, close, volume
        """
        if not self.session:
            raise RuntimeError("Client not initialized. Call initialize() first.")

        self._check_rate_limit()

        url = f'{self.base_url}/v2/aggs/ticker/{symbol}/range/1/{timespan}/{start_date}/{end_date}'
        params = {
            'apiKey': self.api_key,
            'adjusted': 'true' if adjusted else 'false',
            'sort': 'asc',
            'limit': 50000
        }

        all_data = []

        try:
            async with self.session.get(url, params=params, timeout=30) as resp:
                if resp.status == 200:
                    data = await resp.json()

                    if 'results' not in data or not data['results']:
                        logger.warning(f"No data found for {symbol} from {start_date} to {end_date}")
                        return pd.DataFrame()

                    all_data = data['results']

                    # Handle pagination
                    while 'next_url' in data and len(all_data) < 50000:
                        self._check_rate_limit()
                        async with self.session.get(data['next_url'], timeout=30) as next_resp:
                            if next_resp.status == 200:
                                next_data = await next_resp.json()
                                all_data.extend(next_data.get('results', []))
                                data = next_data
                            else:
                                break

                    logger.info(f"Historical data fetched: {symbol} - {len(all_data)} candles")
                    return self._parse_historical(all_data, symbol)
                else:
                    logger.error(f"Historical fetch failed: {symbol} - Status {resp.status}")
                    return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return pd.DataFrame()

    def _parse_historical(self, results: List[Dict], symbol: str) -> pd.DataFrame:
        """Parse historical data response from Polygon API."""
        try:
            df = pd.DataFrame(results)

            # Rename columns to standard format
            column_mapping = {
                'o': 'open',
                'h': 'high',
                'l': 'low',
                'c': 'close',
                'v': 'volume',
                't': 'timestamp',
                'vw': 'vwap',
                'n': 'transactions'
            }

            df = df.rename(columns=column_mapping)

            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            # Keep only OHLCV columns
            keep_columns = ['open', 'high', 'low', 'close', 'volume']
            available_columns = [col for col in keep_columns if col in df.columns]
            df = df[available_columns]

            # Sort by timestamp
            df.sort_index(inplace=True)

            return df
        except Exception as e:
            logger.error(f"Error parsing historical data: {e}")
            return pd.DataFrame()

    async def get_quotes_batch(self, symbols: List[str]) -> Dict[str, Dict]:
        """
        Get quotes for multiple symbols concurrently.

        Args:
            symbols: List of symbols

        Returns:
            Dict mapping symbol to quote data
        """
        tasks = [self.get_quote(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks)
        return {symbols[i]: results[i] for i in range(len(symbols))}

    async def get_technical_indicators(
        self,
        symbol: str,
        indicator: str,
        timespan: str = 'day',
        window: int = 20
    ) -> Dict:
        """
        Get technical indicator values.

        Args:
            symbol: Stock symbol
            indicator: Indicator name (e.g., 'SMA', 'EMA', 'RSI', 'MACD', 'BBANDS')
            timespan: Time period
            window: Window size for calculation

        Returns:
            Dict with indicator data
        """
        if not self.session:
            raise RuntimeError("Client not initialized. Call initialize() first.")

        self._check_rate_limit()

        url = f'{self.base_url}/v1/indicators/{indicator}/{symbol}'
        params = {
            'apiKey': self.api_key,
            'timespan': timespan,
            'window': window,
            'expand_underlying': 'true',
            'order': 'asc',
            'limit': 5000
        }

        try:
            async with self.session.get(url, params=params, timeout=30) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    logger.debug(f"Technical indicator fetched: {symbol} - {indicator}")
                    return data
                else:
                    logger.error(f"Technical indicator fetch failed: {symbol} - Status {resp.status}")
                    return {}
        except Exception as e:
            logger.error(f"Error fetching technical indicator for {symbol}: {e}")
            return {}

    async def test_connection(self) -> Tuple[bool, str]:
        """
        Test API connection and credentials.

        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            quote = await self.get_quote('AAPL')
            if quote and 'price' in quote:
                return True, f"✓ Connection successful. Latest AAPL price: ${quote['price']:.2f}"
            else:
                return False, "✗ Connection failed: No data returned"
        except Exception as e:
            return False, f"✗ Connection failed: {str(e)}"


# Initialize global client
polygon_client = None


async def get_polygon_client() -> PolygonClient:
    """Get or create global Polygon client."""
    global polygon_client
    if polygon_client is None:
        polygon_client = PolygonClient()
        await polygon_client.initialize()
    return polygon_client


async def close_polygon_client():
    """Close global Polygon client."""
    global polygon_client
    if polygon_client:
        await polygon_client.close()
        polygon_client = None
