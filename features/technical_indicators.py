"""
Technical Indicators Module

Comprehensive library of technical indicators for financial analysis.
Includes trend, momentum, volatility, and volume indicators.

Total Indicators: 25+
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """
    Compute technical indicators for financial data.

    Supports:
    - Trend Indicators: SMA, EMA, MACD, ADX
    - Momentum Indicators: RSI, STOCH, CCI, ROC, TRIX
    - Volatility Indicators: ATR, Bollinger Bands, Keltner Channels
    - Volume Indicators: OBV, MFI, AD, VWAP
    """

    @staticmethod
    def sma(df: pd.DataFrame, column: str = 'close', window: int = 20) -> pd.Series:
        """Simple Moving Average"""
        return df[column].rolling(window=window).mean()

    @staticmethod
    def ema(df: pd.DataFrame, column: str = 'close', span: int = 12) -> pd.Series:
        """Exponential Moving Average"""
        return df[column].ewm(span=span, adjust=False).mean()

    @staticmethod
    def macd(df: pd.DataFrame, column: str = 'close',
             fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        MACD (Moving Average Convergence Divergence)

        Returns:
            Tuple of (MACD line, Signal line, Histogram)
        """
        ema_fast = df[column].ewm(span=fast, adjust=False).mean()
        ema_slow = df[column].ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    @staticmethod
    def rsi(df: pd.DataFrame, column: str = 'close', period: int = 14) -> pd.Series:
        """
        Relative Strength Index

        Returns values between 0-100
        """
        delta = df[column].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))

        return rsi

    @staticmethod
    def bollinger_bands(df: pd.DataFrame, column: str = 'close',
                       window: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Bollinger Bands

        Returns:
            Tuple of (Upper Band, Middle Band (SMA), Lower Band)
        """
        middle = df[column].rolling(window=window).mean()
        std = df[column].rolling(window=window).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)

        return upper, middle, lower

    @staticmethod
    def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Average True Range

        Measures volatility
        """
        high = df['high']
        low = df['low']
        close = df['close']

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()

        return atr

    @staticmethod
    def keltner_channels(df: pd.DataFrame, column: str = 'close',
                        window: int = 20, atr_multiplier: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Keltner Channels

        Returns:
            Tuple of (Upper Channel, Middle (EMA), Lower Channel)
        """
        middle = df[column].ewm(span=window, adjust=False).mean()

        high = df['high']
        low = df['low']
        close = df['close']

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=window).mean()

        upper = middle + (atr * atr_multiplier)
        lower = middle - (atr * atr_multiplier)

        return upper, middle, lower

    @staticmethod
    def stochastic(df: pd.DataFrame, period: int = 14,
                   smooth_k: int = 3, smooth_d: int = 3) -> Tuple[pd.Series, pd.Series]:
        """
        Stochastic Oscillator

        Returns:
            Tuple of (K%, D%)
        """
        low_min = df['low'].rolling(window=period).min()
        high_max = df['high'].rolling(window=period).max()

        k_percent = 100 * ((df['close'] - low_min) / (high_max - low_min + 1e-10))
        k_percent = k_percent.rolling(window=smooth_k).mean()
        d_percent = k_percent.rolling(window=smooth_d).mean()

        return k_percent, d_percent

    @staticmethod
    def cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
        """
        Commodity Channel Index

        Measures deviation from average price
        """
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        sma_tp = typical_price.rolling(window=period).mean()
        mad = typical_price.rolling(window=period).apply(
            lambda x: np.mean(np.abs(x - np.mean(x))), raw=False
        )

        cci = (typical_price - sma_tp) / (0.015 * mad + 1e-10)

        return cci

    @staticmethod
    def roc(df: pd.DataFrame, column: str = 'close', period: int = 12) -> pd.Series:
        """
        Rate of Change

        Measures momentum as percentage change
        """
        roc = ((df[column] - df[column].shift(period)) / df[column].shift(period)) * 100

        return roc

    @staticmethod
    def trix(df: pd.DataFrame, column: str = 'close', period: int = 15) -> pd.Series:
        """
        TRIX (Triple Exponential Moving Average)

        Momentum indicator showing percentage change of triple smoothed EMA
        """
        ema1 = df[column].ewm(span=period, adjust=False).mean()
        ema2 = ema1.ewm(span=period, adjust=False).mean()
        ema3 = ema2.ewm(span=period, adjust=False).mean()

        trix = ((ema3 - ema3.shift(1)) / ema3.shift(1)) * 100

        return trix

    @staticmethod
    def obv(df: pd.DataFrame, column: str = 'close') -> pd.Series:
        """
        On-Balance Volume

        Measures positive and negative volume flow
        """
        obv = pd.Series(0, index=df.index)
        obv[0] = df['volume'].iloc[0]

        for i in range(1, len(df)):
            if df[column].iloc[i] > df[column].iloc[i-1]:
                obv[i] = obv[i-1] + df['volume'].iloc[i]
            elif df[column].iloc[i] < df[column].iloc[i-1]:
                obv[i] = obv[i-1] - df['volume'].iloc[i]
            else:
                obv[i] = obv[i-1]

        return obv

    @staticmethod
    def ad(df: pd.DataFrame) -> pd.Series:
        """
        Accumulation/Distribution Line

        Volume-weighted cumulative indicator
        """
        high = df['high']
        low = df['low']
        close = df['close']
        volume = df['volume']

        clv = ((close - low) - (high - close)) / (high - low + 1e-10)
        ad = (clv * volume).cumsum()

        return ad

    @staticmethod
    def mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Money Flow Index

        Volume-weighted RSI
        """
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']

        positive_flow = pd.Series(0, index=df.index)
        negative_flow = pd.Series(0, index=df.index)

        for i in range(1, len(df)):
            if typical_price.iloc[i] > typical_price.iloc[i-1]:
                positive_flow[i] = money_flow.iloc[i]
            else:
                negative_flow[i] = money_flow.iloc[i]

        positive_mf = positive_flow.rolling(window=period).sum()
        negative_mf = negative_flow.rolling(window=period).sum()

        mfi = 100 - (100 / (1 + (positive_mf / (negative_mf + 1e-10))))

        return mfi

    @staticmethod
    def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Average Directional Index

        Measures trend strength
        """
        high = df['high']
        low = df['low']
        close = df['close']

        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Directional Movement
        up_move = high.diff()
        down_move = -low.diff()

        plus_dm = pd.Series(0, index=df.index)
        minus_dm = pd.Series(0, index=df.index)

        for i in range(1, len(df)):
            if up_move.iloc[i] > 0 and up_move.iloc[i] > down_move.iloc[i]:
                plus_dm[i] = up_move.iloc[i]
            if down_move.iloc[i] > 0 and down_move.iloc[i] > up_move.iloc[i]:
                minus_dm[i] = down_move.iloc[i]

        # Smooth
        plus_dm_smooth = plus_dm.rolling(window=period).sum()
        minus_dm_smooth = minus_dm.rolling(window=period).sum()
        tr_smooth = tr.rolling(window=period).sum()

        # DI
        plus_di = 100 * (plus_dm_smooth / (tr_smooth + 1e-10))
        minus_di = 100 * (minus_dm_smooth / (tr_smooth + 1e-10))

        # DX
        di_diff = abs(plus_di - minus_di)
        di_sum = plus_di + minus_di
        dx = 100 * (di_diff / (di_sum + 1e-10))

        # ADX
        adx = dx.rolling(window=period).mean()

        return adx

    @staticmethod
    def vwap(df: pd.DataFrame) -> pd.Series:
        """
        Volume Weighted Average Price
        """
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()

        return vwap

    @staticmethod
    def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all technical indicators for a DataFrame.

        Args:
            df: DataFrame with columns: open, high, low, close, volume

        Returns:
            DataFrame with original OHLCV + all computed indicators
        """
        features = df.copy()

        if len(df) < 30:
            logger.warning(f"Insufficient data ({len(df)} rows) for indicator computation")
            return features

        logger.info(f"Computing indicators for {len(df)} rows")

        try:
            # Trend Indicators
            features['sma_10'] = TechnicalIndicators.sma(df, window=10)
            features['sma_20'] = TechnicalIndicators.sma(df, window=20)
            features['sma_50'] = TechnicalIndicators.sma(df, window=50)
            features['ema_12'] = TechnicalIndicators.ema(df, span=12)
            features['ema_26'] = TechnicalIndicators.ema(df, span=26)

            macd, signal, hist = TechnicalIndicators.macd(df)
            features['macd'] = macd
            features['macd_signal'] = signal
            features['macd_hist'] = hist

            # Momentum Indicators
            features['rsi_14'] = TechnicalIndicators.rsi(df, period=14)

            k, d = TechnicalIndicators.stochastic(df, period=14)
            features['stoch_k'] = k
            features['stoch_d'] = d

            features['cci'] = TechnicalIndicators.cci(df, period=20)
            features['roc_12'] = TechnicalIndicators.roc(df, period=12)
            features['trix'] = TechnicalIndicators.trix(df, period=15)

            # Volatility Indicators
            features['atr'] = TechnicalIndicators.atr(df, period=14)

            bb_upper, bb_middle, bb_lower = TechnicalIndicators.bollinger_bands(df, window=20)
            features['bb_upper'] = bb_upper
            features['bb_middle'] = bb_middle
            features['bb_lower'] = bb_lower
            features['bb_width'] = bb_upper - bb_lower
            features['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower + 1e-10)

            kc_upper, kc_middle, kc_lower = TechnicalIndicators.keltner_channels(df, window=20)
            features['kc_upper'] = kc_upper
            features['kc_middle'] = kc_middle
            features['kc_lower'] = kc_lower

            # Volume Indicators
            features['obv'] = TechnicalIndicators.obv(df)
            features['ad'] = TechnicalIndicators.ad(df)
            features['mfi'] = TechnicalIndicators.mfi(df, period=14)
            features['vwap'] = TechnicalIndicators.vwap(df)

            # Trend Strength
            features['adx'] = TechnicalIndicators.adx(df, period=14)

            # Price-based Features
            features['returns'] = df['close'].pct_change()
            features['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            features['high_low_ratio'] = df['high'] / (df['low'] + 1e-10)
            features['close_open_ratio'] = df['close'] / (df['open'] + 1e-10)

            # Volume-based Features
            features['volume_sma'] = df['volume'].rolling(window=20).mean()
            features['volume_ratio'] = df['volume'] / (df['volume'].rolling(window=20).mean() + 1e-10)

            logger.info(f"Successfully computed {len(features.columns) - len(df.columns)} indicators")

        except Exception as e:
            logger.error(f"Error computing indicators: {e}")
            raise

        return features


# Singleton instance
_indicators_instance = None


def get_indicators() -> TechnicalIndicators:
    """Get or create global technical indicators instance."""
    global _indicators_instance
    if _indicators_instance is None:
        _indicators_instance = TechnicalIndicators()
    return _indicators_instance
