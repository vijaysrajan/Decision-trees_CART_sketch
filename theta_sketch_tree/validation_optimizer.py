"""
Optimized validation data conversion for pruning operations.

This module provides high-performance converters to transform pandas DataFrames
into binary feature matrices for validation-based pruning, with caching support
to avoid repeated conversions.

IMPORTANT: This module does NOT perform feature engineering. It expects DataFrames
with columns that directly map to the feature_mapping provided by the user.
Feature engineering should be done upstream before creating theta sketches.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List, Union
import pickle
import hashlib
import time
from pathlib import Path


class ValidationDataConverter:
    """Optimized converter for DataFrame to binary feature matrix."""

    def __init__(self, cache_dir: Optional[str] = None, enable_cache: bool = True):
        """
        Initialize the converter.

        Parameters
        ----------
        cache_dir : str, optional
            Directory to store conversion cache. If None, uses temporary directory.
        enable_cache : bool, default=True
            Whether to enable caching of conversions.
        """
        self.enable_cache = enable_cache
        self.cache_dir = Path(cache_dir) if cache_dir else Path.cwd() / '.validation_cache'
        self.cache_dir.mkdir(exist_ok=True)

        # Conversion statistics
        self.conversion_stats = {
            'total_conversions': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_time': 0.0,
            'avg_conversion_rate': 0.0,
            'conversion_rate': 0.0
        }

    def _compute_data_hash(self, df: pd.DataFrame, feature_mapping: Dict[str, int]) -> str:
        """Compute hash of DataFrame and feature mapping for caching."""
        # Create hash from DataFrame content and feature mapping
        df_content = pd.util.hash_pandas_object(df, index=True).values
        mapping_str = str(sorted(feature_mapping.items()))

        hasher = hashlib.sha256()
        hasher.update(df_content.tobytes())
        hasher.update(mapping_str.encode())

        return hasher.hexdigest()

    def _load_from_cache(self, cache_key: str) -> Optional[np.ndarray]:
        """Load cached conversion result."""
        if not self.enable_cache:
            return None

        cache_file = self.cache_dir / f"{cache_key}.npy"
        if cache_file.exists():
            try:
                return np.load(cache_file)
            except Exception:
                # Remove corrupted cache file
                cache_file.unlink(missing_ok=True)
                return None
        return None

    def _save_to_cache(self, cache_key: str, data: np.ndarray) -> None:
        """Save conversion result to cache."""
        if not self.enable_cache:
            return

        cache_file = self.cache_dir / f"{cache_key}.npy"
        try:
            np.save(cache_file, data)
        except Exception:
            pass  # Fail silently if cache write fails

    def convert_optimized(self,
                         df: pd.DataFrame,
                         feature_mapping: Dict[str, int],
                         target_col: Optional[str] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Convert DataFrame to binary feature matrix with optimization.

        IMPORTANT: This method expects DataFrames with columns that directly map
        to the provided feature_mapping. No feature engineering is performed.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with binary or numeric columns that map to feature_mapping keys
        feature_mapping : dict
            Mapping from feature names to column indices (must be provided by user)
        target_col : str, optional
            Target column name. If provided, returns separate y array.

        Returns
        -------
        X_binary : np.ndarray
            Binary feature matrix (n_samples, n_features)
        y : np.ndarray or None
            Target array if target_col specified
        """
        start_time = time.time()

        # Generate cache key
        cache_key = self._compute_data_hash(df, feature_mapping)

        # Try to load from cache
        cached_result = self._load_from_cache(cache_key)
        if cached_result is not None:
            self.conversion_stats['cache_hits'] += 1
            self.conversion_stats['total_conversions'] += 1

            # Split cached result if target column was included
            if target_col is not None:
                X_binary = cached_result[:, :-1]
                y = cached_result[:, -1].astype(int)
                return X_binary, y
            else:
                return cached_result, None

        # Cache miss - perform conversion
        self.conversion_stats['cache_misses'] += 1
        X_binary, y = self._convert_dataframe_simple(df, feature_mapping, target_col)

        # Cache the result
        if target_col is not None:
            cache_data = np.column_stack([X_binary, y])
        else:
            cache_data = X_binary
        self._save_to_cache(cache_key, cache_data)

        # Update statistics
        conversion_time = time.time() - start_time
        self.conversion_stats['total_conversions'] += 1
        self.conversion_stats['total_time'] += conversion_time
        if conversion_time > 0:
            self.conversion_stats['avg_conversion_rate'] = len(df) / conversion_time
            self.conversion_stats['conversion_rate'] = len(df) / conversion_time

        return X_binary, y

    def _convert_dataframe_simple(self,
                                 df: pd.DataFrame,
                                 feature_mapping: Dict[str, int],
                                 target_col: Optional[str] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Simple DataFrame to binary matrix conversion - NO feature engineering.

        This method expects the DataFrame columns to directly correspond to
        feature_mapping keys. Any missing features are set to 0.
        """
        n_samples = len(df)
        n_features = len(feature_mapping)

        # Separate target if specified
        if target_col is not None and target_col in df.columns:
            y = df[target_col].values
            df = df.drop(columns=[target_col])
        else:
            y = None

        # Initialize binary matrix
        X_binary = np.zeros((n_samples, n_features), dtype=np.uint8)

        # Simple mapping: expect DataFrame columns to match feature_mapping keys
        for feature_name, feature_idx in feature_mapping.items():
            if feature_name in df.columns:
                # Convert to binary (non-zero = 1, zero/NaN = 0)
                values = df[feature_name].fillna(0)
                X_binary[:, feature_idx] = (values != 0).astype(np.uint8)
            # If feature not found in DataFrame, leave as 0 (missing feature)

        return X_binary, y

    def benchmark_conversion(self,
                           df: pd.DataFrame,
                           feature_mapping: Dict[str, int],
                           target_col: Optional[str] = None,
                           n_repeats: int = 3) -> Dict[str, float]:
        """
        Benchmark conversion performance.

        Returns dictionary with performance metrics.
        """
        times = []

        # Clear cache for fair benchmark
        if self.enable_cache:
            self.clear_cache()

        for _ in range(n_repeats):
            start_time = time.time()
            X_binary, y = self.convert_optimized(df, feature_mapping, target_col)
            end_time = time.time()
            times.append(end_time - start_time)

        avg_time = np.mean(times)
        std_time = np.std(times)
        conversion_rate = len(df) / avg_time if avg_time > 0 else 0

        return {
            'avg_time': avg_time,
            'std_time': std_time,
            'conversion_rate': conversion_rate,
            'samples_per_second': conversion_rate,
            'cache_hit_rate': self.get_cache_hit_rate()
        }

    def clear_cache(self) -> int:
        """Clear conversion cache and return number of files removed."""
        if not self.enable_cache:
            return 0

        removed_count = 0
        for cache_file in self.cache_dir.glob("*.npy"):
            try:
                cache_file.unlink()
                removed_count += 1
            except:
                pass

        return removed_count

    def get_cache_hit_rate(self) -> float:
        """Get cache hit rate as percentage."""
        total = self.conversion_stats['cache_hits'] + self.conversion_stats['cache_misses']
        if total == 0:
            return 0.0
        return (self.conversion_stats['cache_hits'] / total) * 100

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive conversion statistics."""
        stats = self.conversion_stats.copy()
        stats['cache_hit_rate'] = self.get_cache_hit_rate()
        stats['cache_enabled'] = self.enable_cache
        stats['cache_dir'] = str(self.cache_dir)
        return stats


# Convenience function for backward compatibility
def convert_dataframe_to_binary(df: pd.DataFrame,
                               feature_mapping: Dict[str, int],
                               target_col: Optional[str] = None,
                               enable_cache: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Convert DataFrame to binary feature matrix (convenience function).

    IMPORTANT: No feature engineering is performed. DataFrame columns must
    directly correspond to feature_mapping keys.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with columns matching feature_mapping keys
    feature_mapping : dict
        Feature name to index mapping (provided by user)
    target_col : str, optional
        Target column name
    enable_cache : bool, default=True
        Whether to enable caching

    Returns
    -------
    X_binary : np.ndarray
        Binary feature matrix
    y : np.ndarray or None
        Target array if target_col specified
    """
    converter = ValidationDataConverter(enable_cache=enable_cache)
    return converter.convert_optimized(df, feature_mapping, target_col)


def benchmark_conversion_methods(df: pd.DataFrame,
                                feature_mapping: Dict[str, int],
                                target_col: Optional[str] = None) -> Dict[str, Dict[str, float]]:
    """
    Benchmark different conversion methods.

    Returns comparison of cached vs uncached performance.
    """
    results = {}

    # Test with cache enabled
    converter_cached = ValidationDataConverter(enable_cache=True)
    results['cached'] = converter_cached.benchmark_conversion(df, feature_mapping, target_col)

    # Test with cache disabled
    converter_uncached = ValidationDataConverter(enable_cache=False)
    results['uncached'] = converter_uncached.benchmark_conversion(df, feature_mapping, target_col)

    # Calculate speedup
    if results['uncached']['avg_time'] > 0:
        speedup = results['uncached']['avg_time'] / results['cached']['avg_time']
        results['speedup'] = speedup
    else:
        results['speedup'] = 1.0

    return results