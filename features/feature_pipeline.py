"""
Feature Pipeline

Orchestrates feature computation and transformation for machine learning.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging
import json

from features.technical_indicators import TechnicalIndicators
from features.feature_validator import FeatureValidator

logger = logging.getLogger(__name__)


class FeaturePipeline:
    """
    Complete feature engineering pipeline.

    Handles:
    - Feature computation from raw OHLCV
    - Feature validation
    - Feature normalization
    - Window generation for ML
    - Feature persistence
    """

    def __init__(self, output_dir: str = './data/features'):
        self.indicators = TechnicalIndicators()
        self.validator = FeatureValidator()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.output_dir / 'metadata.json'
        self._load_metadata()

    def _load_metadata(self):
        """Load feature metadata."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}

    def _save_metadata(self):
        """Save feature metadata."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def compute_features(self, df: pd.DataFrame, symbol: str = 'UNKNOWN') -> pd.DataFrame:
        """
        Compute all features for a DataFrame.

        Args:
            df: DataFrame with OHLCV data
            symbol: Symbol for logging/tracking

        Returns:
            DataFrame with features computed
        """
        logger.info(f"Computing features for {symbol}: {len(df)} rows")

        if df.empty:
            logger.error(f"Empty DataFrame for {symbol}")
            return df

        # Check minimum requirements
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            logger.error(f"Missing columns for {symbol}: {missing}")
            return df

        try:
            # Compute technical indicators
            features_df = self.indicators.compute_all_indicators(df)
            logger.info(f"Computed {len(features_df.columns) - len(df.columns)} indicators")

            # Validate features
            is_valid, issues = self.validator.validate_features(features_df)
            if not is_valid:
                logger.warning(f"Feature validation issues for {symbol}: {issues}")
                # Repair features
                features_df = self.validator.repair_features(features_df)
                logger.info("Features repaired")

            return features_df

        except Exception as e:
            logger.error(f"Error computing features for {symbol}: {e}")
            raise

    def compute_batch(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Compute features for multiple symbols.

        Args:
            data_dict: Dict mapping symbol to DataFrame

        Returns:
            Dict mapping symbol to features DataFrame
        """
        logger.info(f"Computing features for {len(data_dict)} symbols")

        results = {}
        for symbol, df in data_dict.items():
            try:
                features_df = self.compute_features(df, symbol)
                results[symbol] = features_df

                # Update metadata
                self.metadata[symbol] = {
                    'computed_at': pd.Timestamp.now().isoformat(),
                    'rows': len(features_df),
                    'columns': len(features_df.columns),
                    'features': [col for col in features_df.columns if col not in df.columns]
                }
            except Exception as e:
                logger.error(f"Failed to compute features for {symbol}: {e}")
                results[symbol] = df

        self._save_metadata()
        return results

    def normalize_features(self, df: pd.DataFrame, method: str = 'minmax',
                         exclude_cols: List[str] = None) -> Tuple[pd.DataFrame, Dict]:
        """
        Normalize features.

        Args:
            df: DataFrame with features
            method: 'minmax' (0-1), 'zscore' (mean=0, std=1), 'robust' (median/IQR)
            exclude_cols: Columns to exclude from normalization

        Returns:
            Tuple of (normalized_df, normalization_params)
        """
        if exclude_cols is None:
            exclude_cols = ['open', 'high', 'low', 'close', 'volume']

        cols_to_normalize = [col for col in df.columns if col not in exclude_cols]
        normalized_df = df.copy()
        norm_params = {}

        try:
            if method == 'minmax':
                for col in cols_to_normalize:
                    min_val = df[col].min()
                    max_val = df[col].max()
                    normalized_df[col] = (df[col] - min_val) / (max_val - min_val + 1e-10)
                    norm_params[col] = {'min': float(min_val), 'max': float(max_val), 'method': 'minmax'}

            elif method == 'zscore':
                for col in cols_to_normalize:
                    mean_val = df[col].mean()
                    std_val = df[col].std()
                    normalized_df[col] = (df[col] - mean_val) / (std_val + 1e-10)
                    norm_params[col] = {'mean': float(mean_val), 'std': float(std_val), 'method': 'zscore'}

            elif method == 'robust':
                for col in cols_to_normalize:
                    median_val = df[col].median()
                    q75 = df[col].quantile(0.75)
                    q25 = df[col].quantile(0.25)
                    iqr = q75 - q25
                    normalized_df[col] = (df[col] - median_val) / (iqr + 1e-10)
                    norm_params[col] = {
                        'median': float(median_val),
                        'iqr': float(iqr),
                        'method': 'robust'
                    }

            logger.info(f"Normalized {len(cols_to_normalize)} features using {method}")
            return normalized_df, norm_params

        except Exception as e:
            logger.error(f"Error normalizing features: {e}")
            return df, {}

    def generate_windows(self, df: pd.DataFrame, window_size: int = 60,
                        step_size: int = 1, target_col: str = 'close') -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate sliding windows for ML models.

        Args:
            df: Feature DataFrame
            window_size: Size of each window
            step_size: Step between windows
            target_col: Column to use as target (next day's return)

        Returns:
            Tuple of (X_windows, y_targets)
        """
        if len(df) < window_size + 1:
            logger.error(f"Insufficient data for window generation: {len(df)} < {window_size + 1}")
            return np.array([]), np.array([])

        try:
            X = []
            y = []

            for i in range(0, len(df) - window_size, step_size):
                window = df.iloc[i:i + window_size].values
                target = df[target_col].iloc[i + window_size]
                next_target = df[target_col].iloc[min(i + window_size + 1, len(df) - 1)]

                # Compute return
                target_return = (next_target - target) / target if target > 0 else 0

                X.append(window)
                y.append(target_return)

            X = np.array(X)
            y = np.array(y)

            logger.info(f"Generated {len(X)} windows of size {window_size}")
            return X, y

        except Exception as e:
            logger.error(f"Error generating windows: {e}")
            return np.array([]), np.array([])

    def save_features(self, df: pd.DataFrame, symbol: str, file_format: str = 'parquet') -> bool:
        """
        Save features to disk.

        Args:
            df: Feature DataFrame
            symbol: Symbol for filename
            file_format: 'parquet', 'csv', or 'hdf5'

        Returns:
            Success flag
        """
        try:
            if file_format == 'parquet':
                filepath = self.output_dir / f'{symbol}_features.parquet'
                df.to_parquet(filepath)
            elif file_format == 'csv':
                filepath = self.output_dir / f'{symbol}_features.csv'
                df.to_csv(filepath)
            elif file_format == 'hdf5':
                filepath = self.output_dir / f'{symbol}_features.h5'
                df.to_hdf(filepath, key='features')
            else:
                logger.error(f"Unsupported format: {file_format}")
                return False

            logger.info(f"Saved features for {symbol} to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Error saving features for {symbol}: {e}")
            return False

    def load_features(self, symbol: str, file_format: str = 'parquet') -> Optional[pd.DataFrame]:
        """
        Load features from disk.

        Args:
            symbol: Symbol to load
            file_format: 'parquet', 'csv', or 'hdf5'

        Returns:
            Feature DataFrame or None
        """
        try:
            if file_format == 'parquet':
                filepath = self.output_dir / f'{symbol}_features.parquet'
                if filepath.exists():
                    return pd.read_parquet(filepath)
            elif file_format == 'csv':
                filepath = self.output_dir / f'{symbol}_features.csv'
                if filepath.exists():
                    return pd.read_csv(filepath, index_col=0)
            elif file_format == 'hdf5':
                filepath = self.output_dir / f'{symbol}_features.h5'
                if filepath.exists():
                    return pd.read_hdf(filepath, key='features')

            logger.warning(f"Feature file not found for {symbol}")
            return None

        except Exception as e:
            logger.error(f"Error loading features for {symbol}: {e}")
            return None

    def generate_report(self, df: pd.DataFrame, symbol: str = 'Unknown') -> Dict:
        """
        Generate comprehensive feature report.

        Args:
            df: Feature DataFrame
            symbol: Symbol for reporting

        Returns:
            Dict with feature statistics
        """
        report = {
            'symbol': symbol,
            'timestamp': pd.Timestamp.now().isoformat(),
            'data_quality': {
                'rows': len(df),
                'columns': len(df.columns),
                'nan_count': int(df.isna().sum().sum()),
                'inf_count': int(np.isinf(df.values).sum())
            },
            'feature_statistics': {}
        }

        # Compute stats for each numeric column
        for col in df.select_dtypes(include=[np.number]).columns:
            report['feature_statistics'][col] = {
                'mean': float(df[col].mean()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'median': float(df[col].median()),
                'q25': float(df[col].quantile(0.25)),
                'q75': float(df[col].quantile(0.75))
            }

        return report

    def get_pipeline_stats(self) -> Dict:
        """Get overall pipeline statistics."""
        return {
            'features_dir': str(self.output_dir),
            'computed_symbols': len(self.metadata),
            'symbols': list(self.metadata.keys()),
            'metadata': self.metadata
        }


# Singleton instance
_pipeline_instance = None


async def get_pipeline() -> FeaturePipeline:
    """Get or create global pipeline instance."""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = FeaturePipeline()
    return _pipeline_instance


async def close_pipeline():
    """Close global pipeline instance."""
    global _pipeline_instance
    if _pipeline_instance:
        _pipeline_instance = None
