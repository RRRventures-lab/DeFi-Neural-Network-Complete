"""
Feature Validator

Validates computed features for quality and integrity.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
import logging

logger = logging.getLogger(__name__)


class FeatureValidator:
    """
    Validate computed features.

    Checks for:
    - NaN and infinite values
    - Feature correlation issues
    - Outliers
    - Data type consistency
    """

    def __init__(self, max_nan_percentage: float = 0.1, max_inf_percentage: float = 0.01):
        self.max_nan_percentage = max_nan_percentage
        self.max_inf_percentage = max_inf_percentage

    def validate_features(self, df: pd.DataFrame) -> Tuple[bool, Dict]:
        """
        Validate feature DataFrame.

        Args:
            df: DataFrame with computed features

        Returns:
            Tuple of (is_valid, issues)
        """
        issues = {}

        # Check for empty DataFrame
        if df.empty:
            issues['empty_dataframe'] = True
            return False, issues

        # Check for NaN values
        nan_counts = df.isna().sum()
        nan_percentage = nan_counts.sum() / (len(df) * len(df.columns))

        if nan_percentage > self.max_nan_percentage:
            issues['high_nan_percentage'] = float(nan_percentage)
            nan_by_column = nan_counts[nan_counts > 0].to_dict()
            if nan_by_column:
                issues['nan_by_column'] = {k: int(v) for k, v in nan_by_column.items()}

        # Check for infinite values
        inf_mask = np.isinf(df.values)
        inf_count = np.sum(inf_mask)

        if inf_count > 0:
            inf_percentage = inf_count / (len(df) * len(df.columns))
            if inf_percentage > self.max_inf_percentage:
                issues['high_inf_percentage'] = float(inf_percentage)

        # Check for all-zero or constant columns
        numeric_df = df.select_dtypes(include=[np.number])
        for col in numeric_df.columns:
            if numeric_df[col].std() == 0:
                issues[f'constant_column_{col}'] = True

        # Check for extreme outliers
        for col in numeric_df.columns:
            z_scores = np.abs((numeric_df[col] - numeric_df[col].mean()) / (numeric_df[col].std() + 1e-10))
            extreme_outliers = (z_scores > 5).sum()
            if extreme_outliers > 0:
                outlier_percentage = extreme_outliers / len(df)
                if outlier_percentage > 0.05:  # More than 5%
                    issues[f'extreme_outliers_{col}'] = int(extreme_outliers)

        # Log issues
        if issues:
            logger.warning(f"Feature validation issues found: {issues}")
            return False, issues
        else:
            logger.info("Feature validation passed")
            return True, {}

    def repair_features(self, df: pd.DataFrame, method: str = 'forward_fill') -> pd.DataFrame:
        """
        Repair common feature issues.

        Args:
            df: DataFrame with potential issues
            method: 'forward_fill', 'backward_fill', 'interpolate', 'zero'

        Returns:
            Repaired DataFrame
        """
        df = df.copy()

        # Replace infinities with NaN
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

        # Fill NaN values
        if method == 'forward_fill':
            df = df.fillna(method='ffill').fillna(method='bfill')
        elif method == 'backward_fill':
            df = df.fillna(method='bfill').fillna(method='ffill')
        elif method == 'interpolate':
            df[numeric_cols] = df[numeric_cols].interpolate(method='linear', limit_direction='both')
        elif method == 'zero':
            df = df.fillna(0)

        # Handle any remaining NaN (edge cases)
        remaining_nan = df.isna().sum().sum()
        if remaining_nan > 0:
            logger.info(f"Filling remaining {remaining_nan} NaN values with 0")
            df = df.fillna(0)

        # Handle any remaining infinities (shouldn't happen but be safe)
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], 0)

        logger.info(f"Feature repair completed using {method} method")
        return df

    def check_feature_correlation(self, df: pd.DataFrame, threshold: float = 0.99) -> Dict[str, List[Tuple]]:
        """
        Detect highly correlated features.

        Args:
            df: Feature DataFrame
            threshold: Correlation threshold for high correlation

        Returns:
            Dict mapping feature to list of correlated features
        """
        numeric_df = df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr().abs()

        high_corr = {}

        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                if corr_matrix.iloc[i, j] > threshold:
                    col_i = corr_matrix.columns[i]
                    col_j = corr_matrix.columns[j]
                    corr_value = float(corr_matrix.iloc[i, j])

                    if col_i not in high_corr:
                        high_corr[col_i] = []
                    high_corr[col_i].append((col_j, corr_value))

        if high_corr:
            logger.warning(f"Found {len(high_corr)} highly correlated feature pairs")

        return high_corr

    def detect_outliers(self, df: pd.DataFrame, threshold: float = 3.0) -> Dict[str, int]:
        """
        Detect outliers using z-score.

        Args:
            df: Feature DataFrame
            threshold: Z-score threshold

        Returns:
            Dict mapping column to number of outliers
        """
        numeric_df = df.select_dtypes(include=[np.number])
        outliers = {}

        for col in numeric_df.columns:
            z_scores = np.abs((numeric_df[col] - numeric_df[col].mean()) / (numeric_df[col].std() + 1e-10))
            outlier_count = (z_scores > threshold).sum()

            if outlier_count > 0:
                outliers[col] = int(outlier_count)

        if outliers:
            logger.info(f"Detected {len(outliers)} columns with outliers")

        return outliers

    def check_feature_completeness(self, df: pd.DataFrame, min_rows: int = 100) -> Dict:
        """
        Check if features are complete and sufficient.

        Args:
            df: Feature DataFrame
            min_rows: Minimum required rows

        Returns:
            Dict with completeness metrics
        """
        numeric_df = df.select_dtypes(include=[np.number])

        return {
            'rows': len(df),
            'columns': len(df.columns),
            'numeric_columns': len(numeric_df.columns),
            'min_rows_met': len(df) >= min_rows,
            'nan_percentage': float(df.isna().sum().sum() / (len(df) * len(df.columns))),
            'complete_rows': int((~df.isna().any(axis=1)).sum())
        }

    def get_feature_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Generate comprehensive feature statistics.

        Args:
            df: Feature DataFrame

        Returns:
            Dict with statistics for each feature
        """
        stats = {}
        numeric_df = df.select_dtypes(include=[np.number])

        for col in numeric_df.columns:
            stats[col] = {
                'dtype': str(numeric_df[col].dtype),
                'count': int(numeric_df[col].count()),
                'mean': float(numeric_df[col].mean()),
                'std': float(numeric_df[col].std()),
                'min': float(numeric_df[col].min()),
                'q25': float(numeric_df[col].quantile(0.25)),
                'median': float(numeric_df[col].median()),
                'q75': float(numeric_df[col].quantile(0.75)),
                'max': float(numeric_df[col].max()),
                'skewness': float(numeric_df[col].skew()),
                'kurtosis': float(numeric_df[col].kurtosis())
            }

        return stats

    def get_feature_names(self, df: pd.DataFrame, exclude_ohlcv: bool = True) -> List[str]:
        """
        Get list of computed features.

        Args:
            df: Feature DataFrame
            exclude_ohlcv: Exclude OHLCV columns

        Returns:
            List of feature names
        """
        ohlcv = ['open', 'high', 'low', 'close', 'volume']
        feature_names = []

        for col in df.columns:
            if exclude_ohlcv and col in ohlcv:
                continue
            feature_names.append(col)

        return feature_names

    def compare_features(self, df1: pd.DataFrame, df2: pd.DataFrame) -> Dict:
        """
        Compare two feature DataFrames.

        Args:
            df1: First DataFrame
            df2: Second DataFrame

        Returns:
            Comparison results
        """
        comparison = {
            'shapes_equal': df1.shape == df2.shape,
            'columns_equal': list(df1.columns) == list(df2.columns),
            'index_equal': df1.index.equals(df2.index)
        }

        if comparison['columns_equal']:
            numeric_df1 = df1.select_dtypes(include=[np.number])
            numeric_df2 = df2.select_dtypes(include=[np.number])

            for col in numeric_df1.columns:
                if col in numeric_df2.columns:
                    diff = (numeric_df1[col] - numeric_df2[col]).abs().sum()
                    comparison[f'{col}_diff'] = float(diff)

        return comparison


# Singleton instance
_validator_instance = None


def get_validator() -> FeatureValidator:
    """Get or create global validator instance."""
    global _validator_instance
    if _validator_instance is None:
        _validator_instance = FeatureValidator()
    return _validator_instance
