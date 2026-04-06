from __future__ import annotations
"""
Regime Detector — HMM-based war regime classifier (Module 1).

Classifies each monthly snapshot into one of three war regimes:
  - Escalation (0)
  - Plateau (1)
  - De-escalation (2)

Uses a Hidden Markov Model (hmmlearn) trained on:
  - Oil volatility
  - ACLED event count (normalized)
  - GDELT tone score (average)
"""

import numpy as np
import pandas as pd

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import N_REGIMES, REGIME_LABELS
from utils.logger import get_logger, log_step

logger = get_logger('model.regime_detector')

# Check if hmmlearn is available
try:
    from hmmlearn.hmm import GaussianHMM
    HAS_HMM = True
except ImportError:
    logger.warning('hmmlearn not installed, using threshold-based fallback')
    HAS_HMM = False


class RegimeDetector:
    """
    HMM-based war regime classifier.
    
    Identifies market regimes (Escalation, Plateau, De-escalation)
    from geopolitical indicators to condition GNN predictions.
    """

    def __init__(self, n_regimes: int = N_REGIMES):
        self.n_regimes = n_regimes
        self.labels = REGIME_LABELS

        if HAS_HMM:
            self.model = GaussianHMM(
                n_components=n_regimes,
                covariance_type='full',
                n_iter=200,
                random_state=42
            )
        else:
            self.model = None

        self._fitted = False

    def fit(self, features: np.ndarray) -> 'RegimeDetector':
        """
        Fit the HMM on regime indicator features.
        
        Args:
            features: Array of shape (T, 3)
                      Columns: [oil_vol, acled_count_norm, gdelt_tone]
                      
        Returns:
            self
        """
        log_step(logger, 'Fitting regime detector',
                 f'Samples={features.shape[0]}, features={features.shape[1]}')

        if HAS_HMM:
            self.model.fit(features)
            self._fitted = True
            logger.info(f'HMM fitted with {self.n_regimes} regimes')
            logger.info(f'Transition matrix:\n{self.model.transmat_.round(3)}')
        else:
            self._fitted = True
            logger.info('Using threshold-based regime detection (hmmlearn not available)')

        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predict regime labels for each time step.
        
        Args:
            features: Array of shape (T, 3)
            
        Returns:
            Array of regime labels (T,)
        """
        if HAS_HMM and self._fitted:
            return self.model.predict(features)
        else:
            return self._threshold_predict(features)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """
        Get regime probability distributions.
        
        Returns:
            Array of shape (T, n_regimes) with probabilities
        """
        if HAS_HMM and self._fitted:
            return self.model.predict_proba(features)
        else:
            labels = self._threshold_predict(features)
            proba = np.zeros((len(labels), self.n_regimes))
            for i, l in enumerate(labels):
                proba[i, l] = 0.7
                for j in range(self.n_regimes):
                    if j != l:
                        proba[i, j] = 0.3 / (self.n_regimes - 1)
            return proba

    def _threshold_predict(self, features: np.ndarray) -> np.ndarray:
        """
        Simple threshold-based regime detection fallback.
        Uses composite score of all indicators.
        """
        # Composite escalation score
        # Higher oil vol + higher event count + more negative tone = escalation
        composite = features[:, 0] + features[:, 1] - features[:, 2]

        # Normalize
        mean = composite.mean()
        std = composite.std() if composite.std() > 0 else 1.0
        z = (composite - mean) / std

        # Classify
        labels = np.ones(len(z), dtype=int)  # Default: Plateau
        labels[z > 0.5] = 0   # Escalation
        labels[z < -0.5] = 2  # De-escalation

        return labels

    def get_regime_label(self, regime_id: int) -> str:
        """Get human-readable regime label."""
        return self.labels.get(regime_id, f'Unknown({regime_id})')

    def get_current_regime(self, features: np.ndarray) -> tuple[int, str]:
        """Get the current (latest) regime."""
        labels = self.predict(features)
        current = labels[-1]
        return current, self.get_regime_label(current)


def build_regime_features(oil_prices: pd.Series,
                           events_df: pd.DataFrame,
                           gdelt_df: pd.DataFrame = None,
                           window: int = 30) -> np.ndarray:
    """
    Build regime indicator features from raw data.
    
    Computes:
    1. Oil volatility (rolling std of oil returns)
    2. ACLED event count (monthly, normalized)
    3. GDELT average tone score
    
    Args:
        oil_prices: Daily oil price series
        events_df: ACLED events DataFrame
        gdelt_df: GDELT headlines DataFrame (optional)
        window: Rolling window for oil vol
        
    Returns:
        numpy array of shape (T_months, 3)
    """
    log_step(logger, 'Building regime indicator features')

    # 1. Monthly oil volatility
    oil_rets = oil_prices.pct_change().dropna()
    oil_vol = oil_rets.rolling(window).std() * np.sqrt(252)
    oil_vol_monthly = oil_vol.resample('ME').last().dropna()

    # 2. Monthly ACLED event count (normalized)
    events_df = events_df.copy()
    events_df['month'] = events_df['event_date'].dt.to_period('M')
    event_counts = events_df.groupby('month').size()
    event_counts.index = event_counts.index.to_timestamp()
    # Normalize by max
    event_counts_norm = event_counts / (event_counts.max() + 1e-8)

    # 3. Monthly GDELT tone score
    if gdelt_df is not None and 'tone_score' in gdelt_df.columns:
        gdelt_df = gdelt_df.copy()
        gdelt_df['month'] = gdelt_df['event_date'].dt.to_period('M')
        tone = gdelt_df.groupby('month')['tone_score'].mean()
        tone.index = tone.index.to_timestamp()
    else:
        tone = pd.Series(0.0, index=oil_vol_monthly.index)

    # Align all indices
    common_idx = oil_vol_monthly.index
    event_reindexed = event_counts_norm.reindex(common_idx).fillna(0)
    tone_reindexed = tone.reindex(common_idx).fillna(0)

    features = np.column_stack([
        oil_vol_monthly.values,
        event_reindexed.values,
        tone_reindexed.values
    ])

    logger.info(f'Regime features shape: {features.shape}')
    return features


if __name__ == '__main__':
    # Test with synthetic data
    np.random.seed(42)
    T = 20
    features = np.column_stack([
        np.random.randn(T) * 0.3 + 0.2,   # oil vol
        np.random.rand(T),                  # event count
        np.random.randn(T) * 3,             # tone
    ])

    detector = RegimeDetector()
    detector.fit(features)
    labels = detector.predict(features)
    print(f'Regime labels: {labels}')
    print(f'Labels: {[detector.get_regime_label(l) for l in labels]}')
