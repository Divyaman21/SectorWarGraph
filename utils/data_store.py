from __future__ import annotations
"""
Parquet-based data caching utility.
Saves and loads DataFrames to/from parquet files to avoid redundant API calls.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

from utils.logger import get_logger

logger = get_logger('data_store')


class DataStore:
    """Parquet-based data cache with expiration support."""

    def __init__(self, cache_dir: str = 'data/cache'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f'DataStore initialized at {self.cache_dir.resolve()}')

    def _path(self, key: str) -> Path:
        """Get the file path for a cache key."""
        safe_key = key.replace('/', '_').replace('\\', '_')
        return self.cache_dir / f'{safe_key}.parquet'

    def _meta_path(self, key: str) -> Path:
        """Get the metadata path for a cache key."""
        safe_key = key.replace('/', '_').replace('\\', '_')
        return self.cache_dir / f'{safe_key}.meta'

    def save(self, key: str, df: pd.DataFrame) -> None:
        """Save a DataFrame to parquet cache."""
        path = self._path(key)
        df.to_parquet(path, engine='pyarrow')
        # Write timestamp metadata
        meta = self._meta_path(key)
        meta.write_text(datetime.now().isoformat())
        logger.info(f'Cached {key}: {df.shape[0]} rows → {path.name}')

    def load(self, key: str, max_age_hours: float = 24.0) -> pd.DataFrame | None:
        """
        Load a DataFrame from parquet cache if it exists and is fresh.
        
        Args:
            key: Cache key
            max_age_hours: Maximum age in hours before cache is stale
            
        Returns:
            DataFrame if cache is valid, None otherwise
        """
        path = self._path(key)
        meta = self._meta_path(key)
        
        if not path.exists():
            return None
        
        if meta.exists():
            cached_time = datetime.fromisoformat(meta.read_text().strip())
            age = datetime.now() - cached_time
            if age > timedelta(hours=max_age_hours):
                logger.info(f'Cache expired for {key} (age={age})')
                return None
        
        df = pd.read_parquet(path, engine='pyarrow')
        logger.info(f'Cache hit for {key}: {df.shape[0]} rows')
        return df

    def save_numpy(self, key: str, arr: np.ndarray) -> None:
        """Save a numpy array to cache."""
        path = self.cache_dir / f'{key}.npy'
        np.save(path, arr)
        meta = self._meta_path(key)
        meta.write_text(datetime.now().isoformat())
        logger.info(f'Cached numpy {key}: shape={arr.shape}')

    def load_numpy(self, key: str, max_age_hours: float = 24.0) -> np.ndarray | None:
        """Load a numpy array from cache."""
        path = self.cache_dir / f'{key}.npy'
        meta = self._meta_path(key)
        
        if not path.exists():
            return None
        
        if meta.exists():
            cached_time = datetime.fromisoformat(meta.read_text().strip())
            age = datetime.now() - cached_time
            if age > timedelta(hours=max_age_hours):
                return None
        
        arr = np.load(path)
        logger.info(f'Cache hit for numpy {key}: shape={arr.shape}')
        return arr

    def clear(self, key: str = None) -> None:
        """Clear a specific cache key or all cached data."""
        if key:
            for p in [self._path(key), self._meta_path(key)]:
                if p.exists():
                    p.unlink()
            logger.info(f'Cleared cache for {key}')
        else:
            for f in self.cache_dir.glob('*'):
                f.unlink()
            logger.info('Cleared all cache')
