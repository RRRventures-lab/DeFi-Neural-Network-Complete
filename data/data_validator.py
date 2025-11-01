"""
Data Validation Module

Validates OHLCV data quality and detects/repairs common issues.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
import logging

logger = logging.getLogger(__name__)


class DataValidator:
    """
    Validates financial OHLCV data for quality issues.

    Checks for:
    - Missing columns
    - Invalid price relationships (high < low, etc.)
    - Extreme price movements (likely data errors)
    - Trading day gaps
    - Volume anomalies
    - NaN/infinite values
    """

    def __init__(self):
        self.max_daily_return = 0.50  # 50% max move detected as potential error
        self.min_volume_percentile = 10  # Flag if volume is in bottom 10%

    def validate_ohlcv(self, df: pd.DataFrame) -> Tuple[bool, Dict]:
        """
        Validate OHLCV DataFrame.

        Args:
            df: DataFrame with columns: open, high, low, close, volume

        Returns:
            Tuple of (is_valid: bool, issues: Dict)
        """
        issues = {}

        # Check columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            issues['missing_columns'] = missing_cols
            return False, issues

        # Check data types
        for col in required_cols:
            if not pd.api.types.is_numeric_dtype(df[col]):
                issues['invalid_dtype'] = {col: str(df[col].dtype)}

        # Check for NaN values
        nan_counts = df[required_cols].isna().sum()
        if nan_counts.any():
            issues['nan_values'] = nan_counts.to_dict()

        # Check for infinite values
        inf_counts = {}
        for col in required_cols:
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                inf_counts[col] = inf_count
        if inf_counts:
            issues['infinite_values'] = inf_counts

        # Check price relationships (high >= low, open/close within range)
        invalid_high_low = (df['high'] < df['low']).sum()
        if invalid_high_low > 0:
            issues['high_less_than_low'] = invalid_high_low

        invalid_close_high = (df['close'] > df['high']).sum()
        if invalid_close_high > 0:
            issues['close_above_high'] = invalid_close_high

        invalid_close_low = (df['close'] < df['low']).sum()
        if invalid_close_low > 0:
            issues['close_below_low'] = invalid_close_low

        invalid_open_high = (df['open'] > df['high']).sum()
        if invalid_open_high > 0:
            issues['open_above_high'] = invalid_open_high

        invalid_open_low = (df['open'] < df['low']).sum()
        if invalid_open_low > 0:
            issues['open_below_low'] = invalid_open_low

        # Check for extreme returns (likely data errors)
        returns = df['close'].pct_change()
        extreme_moves = (np.abs(returns) > self.max_daily_return).sum()
        if extreme_moves > 0:
            issues['extreme_returns'] = {
                'count': extreme_moves,
                'threshold': self.max_daily_return
            }

        # Check for trading day gaps (weekends/holidays acceptable)
        if len(df) > 1:
            time_diffs = df.index.to_series().diff()
            # More than 3 business days
            large_gaps = (time_diffs > pd.Timedelta(days=3)).sum()
            if large_gaps > 0:
                issues['large_time_gaps'] = large_gaps

        # Check volume
        if df['volume'].sum() == 0:
            issues['zero_volume'] = len(df)

        zero_volume_rows = (df['volume'] == 0).sum()
        if zero_volume_rows > len(df) * 0.1:  # More than 10% zero volume
            issues['high_zero_volume_percentage'] = zero_volume_rows / len(df)

        # Check for duplicate timestamps
        if df.index.duplicated().any():
            issues['duplicate_timestamps'] = df.index.duplicated().sum()

        # Log issues
        if issues:
            logger.warning(f"Data validation issues found: {issues}")
            return False, issues
        else:
            logger.info("Data validation passed")
            return True, {}

    def repair_ohlcv(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Repair common OHLCV data issues.

        Repairs:
        - NaN values (forward fill, then back fill)
        - Infinite values (replace with NaN then fill)
        - High < Low (adjust both to be correct)
        - Close outside range (clip to high/low)
        - Duplicate timestamps (keep first)

        Args:
            df: Potentially problematic DataFrame

        Returns:
            Repaired DataFrame
        """
        df = df.copy()

        required_cols = ['open', 'high', 'low', 'close', 'volume']

        # Remove duplicates (keep first)
        if df.index.duplicated().any():
            logger.info(f"Removing {df.index.duplicated().sum()} duplicate timestamps")
            df = df[~df.index.duplicated(keep='first')]

        # Replace infinities with NaN
        for col in required_cols:
            if col in df.columns:
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)

        # Forward fill then backward fill NaN values
        for col in required_cols:
            if col in df.columns:
                nan_count = df[col].isna().sum()
                if nan_count > 0:
                    logger.info(f"Filling {nan_count} NaN values in {col}")
                    df[col] = df[col].fillna(method='ffill').fillna(method='bfill')

        # Ensure high >= low
        if 'high' in df.columns and 'low' in df.columns:
            mask = df['high'] < df['low']
            if mask.any():
                logger.info(f"Fixing {mask.sum()} rows where high < low")
                df.loc[mask, 'high'], df.loc[mask, 'low'] = (
                    df.loc[mask, 'low'], df.loc[mask, 'high']
                )

        # Clip close to [low, high]
        if all(col in df.columns for col in ['close', 'high', 'low']):
            df['close'] = df['close'].clip(lower=df['low'], upper=df['high'])

        # Clip open to [low, high]
        if all(col in df.columns for col in ['open', 'high', 'low']):
            df['open'] = df['open'].clip(lower=df['low'], upper=df['high'])

        # Ensure non-negative volume
        if 'volume' in df.columns:
            df['volume'] = df['volume'].clip(lower=0)

        logger.info("Data repair completed")
        return df

    def detect_outliers(self, df: pd.DataFrame, column: str = 'close', std_threshold: float = 3.0) -> pd.DataFrame:
        """
        Detect statistical outliers using z-score.

        Args:
            df: DataFrame
            column: Column to check for outliers
            std_threshold: Number of standard deviations for outlier threshold

        Returns:
            DataFrame with 'is_outlier' column
        """
        if column not in df.columns:
            raise ValueError(f"Column {column} not found in DataFrame")

        returns = df[column].pct_change()
        z_scores = np.abs((returns - returns.mean()) / returns.std())
        df['is_outlier'] = z_scores > std_threshold

        outlier_count = df['is_outlier'].sum()
        if outlier_count > 0:
            logger.info(f"Detected {outlier_count} outliers in {column}")

        return df

    def check_data_freshness(self, df: pd.DataFrame, max_age_days: int = 30) -> Tuple[bool, Dict]:
        """
        Check if data is recent enough.

        Args:
            df: DataFrame
            max_age_days: Maximum age in days

        Returns:
            Tuple of (is_fresh: bool, info: Dict)
        """
        if df.empty:
            return False, {'reason': 'Empty DataFrame'}

        latest_date = df.index.max()
        age_days = (pd.Timestamp.now() - latest_date).days

        is_fresh = age_days <= max_age_days

        return is_fresh, {
            'latest_date': latest_date.strftime('%Y-%m-%d'),
            'age_days': age_days,
            'is_fresh': is_fresh
        }

    def check_sufficient_data(self, df: pd.DataFrame, min_rows: int = 100) -> Tuple[bool, Dict]:
        """
        Check if DataFrame has sufficient data.

        Args:
            df: DataFrame
            min_rows: Minimum required rows

        Returns:
            Tuple of (is_sufficient: bool, info: Dict)
        """
        is_sufficient = len(df) >= min_rows

        return is_sufficient, {
            'rows': len(df),
            'min_required': min_rows,
            'is_sufficient': is_sufficient
        }

    def compare_dataframes(self, df1: pd.DataFrame, df2: pd.DataFrame) -> Dict:
        """
        Compare two DataFrames for consistency.

        Args:
            df1: First DataFrame
            df2: Second DataFrame

        Returns:
            Dict with comparison results
        """
        comparison = {
            'shape_equal': df1.shape == df2.shape,
            'columns_equal': list(df1.columns) == list(df2.columns),
            'index_equal': df1.index.equals(df2.index),
        }

        if comparison['columns_equal']:
            for col in df1.columns:
                comparison[f'{col}_equal'] = df1[col].equals(df2[col])
                if not comparison[f'{col}_equal']:
                    diff = (df1[col] - df2[col]).abs().sum()
                    comparison[f'{col}_difference'] = float(diff)

        return comparison

    def generate_quality_report(self, df: pd.DataFrame, symbol: str = 'Unknown') -> Dict:
        """
        Generate comprehensive data quality report.

        Args:
            df: DataFrame to analyze
            symbol: Symbol for reporting

        Returns:
            Dict with quality metrics
        """
        valid, issues = self.validate_ohlcv(df)
        is_fresh, freshness = self.check_data_freshness(df)
        is_sufficient, sufficiency = self.check_sufficient_data(df)

        report = {
            'symbol': symbol,
            'timestamp': pd.Timestamp.now().isoformat(),
            'validation': {
                'is_valid': valid,
                'issues': issues
            },
            'freshness': freshness,
            'sufficiency': sufficiency,
            'statistics': {
                'rows': len(df),
                'date_range': f"{df.index.min()} to {df.index.max()}",
                'avg_volume': float(df['volume'].mean()) if 'volume' in df.columns else None,
                'price_range': {
                    'min': float(df['close'].min()) if 'close' in df.columns else None,
                    'max': float(df['close'].max()) if 'close' in df.columns else None,
                    'std': float(df['close'].std()) if 'close' in df.columns else None,
                }
            }
        }

        return report


# Singleton instance
_validator_instance = None


def get_validator() -> DataValidator:
    """Get or create global validator instance."""
    global _validator_instance
    if _validator_instance is None:
        _validator_instance = DataValidator()
    return _validator_instance
