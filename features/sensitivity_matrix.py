from __future__ import annotations
"""
Sector Sensitivity Matrix — THE KEY FILE.
Bridge between raw geopolitical events and sector-level financial impact.

Shape: (N_event_types x N_sectors)
Each cell: signed float representing how strongly a sector reacts to a war event type.
Positive = shock amplifies returns/vol, Negative = sector benefits.

Three-phase build strategy:
  Phase 1: Hand-crafted prior (domain knowledge)
  Phase 2: Empirical calibration from historical data
  Phase 3: Learned sensitivity via Lasso regression
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import MultiTaskLasso
from sklearn.preprocessing import StandardScaler

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import SECTOR_ETFS, LASSO_ALPHA
from utils.logger import get_logger, log_step

logger = get_logger('features.sensitivity_matrix')

SECTORS = list(SECTOR_ETFS.keys())
# ['XLC', 'XLY', 'XLP', 'XLF', 'XLE', 'XLV', 'XLI', 'XLK', 'XLB', 'XLRE', 'XLU']

# ══════════════════════════════════════════════════════════════════════════════
# Phase 1: Hand-Crafted Prior Matrix (15 event types x 11 sectors)
# ══════════════════════════════════════════════════════════════════════════════
#                                   XLC   XLY   XLP   XLF   XLE   XLV   XLI   XLK   XLB   XLRE  XLU
EVENT_TYPES = {
    'oil_route_threat':      [ 0.1,  0.3,  0.2,  0.1,  0.9,  0.0,  0.6,  0.0,  0.7,  0.1,  0.5],
    'oil_price_spike':       [ 0.0,  0.2,  0.2,  0.1,  1.0,  0.0,  0.5,  0.0,  0.8,  0.0,  0.4],
    'military_strike_MENA':  [ 0.2,  0.1,  0.1,  0.3,  0.6,  0.1,  0.3,  0.1,  0.4,  0.1,  0.2],
    'shipping_disruption':   [ 0.0,  0.5,  0.4,  0.1,  0.4,  0.0,  0.8,  0.1,  0.6,  0.0,  0.1],
    'us_iran_tension':       [ 0.1,  0.2,  0.1,  0.4,  0.8,  0.0,  0.4,  0.2,  0.5,  0.2,  0.3],
    'ceasefire_signal':      [ 0.1, -0.1,  0.0,  0.0, -0.5,  0.0, -0.3,  0.1, -0.4,  0.1, -0.3],
    'sanctions_imposed':     [ 0.0,  0.1,  0.0,  0.5,  0.6,  0.0,  0.2,  0.3,  0.4,  0.1,  0.1],
    'humanitarian_crisis':   [ 0.3,  0.0,  0.2,  0.0,  0.0,  0.4,  0.0,  0.1,  0.0,  0.0,  0.0],
    'cyber_attack_MENA':     [ 0.5,  0.1,  0.0,  0.3,  0.1,  0.1,  0.2,  0.6,  0.0,  0.0,  0.1],
    'houthi_missile':        [ 0.0,  0.3,  0.2,  0.1,  0.7,  0.0,  0.7,  0.0,  0.5,  0.0,  0.2],
    'iran_nuclear_progress': [ 0.0,  0.1,  0.0,  0.2,  0.9,  0.0,  0.3,  0.1,  0.5,  0.1,  0.4],
    'israel_ground_op':      [ 0.2,  0.1,  0.1,  0.2,  0.5,  0.2,  0.2,  0.1,  0.3,  0.1,  0.1],
    'hezbollah_escalation':  [ 0.1,  0.2,  0.1,  0.2,  0.7,  0.0,  0.4,  0.1,  0.4,  0.1,  0.3],
    'opec_cut_announcement': [ 0.0,  0.1,  0.1,  0.0,  1.0,  0.0,  0.3,  0.0,  0.6,  0.0,  0.3],
    'diplomatic_progress':   [ 0.1, -0.1,  0.0,  0.1, -0.3,  0.0, -0.2,  0.1, -0.2,  0.1, -0.2],
}

# Build numpy matrix: shape (15, 11)
SENSITIVITY_MATRIX = np.array(list(EVENT_TYPES.values()))
SENSITIVITY_DF = pd.DataFrame(
    SENSITIVITY_MATRIX,
    index=list(EVENT_TYPES.keys()),
    columns=SECTORS
)


def get_sensitivity_matrix() -> pd.DataFrame:
    """Return a copy of the current sensitivity matrix."""
    return SENSITIVITY_DF.copy()


def get_event_types() -> list[str]:
    """Return list of all event type names."""
    return list(EVENT_TYPES.keys())


# ══════════════════════════════════════════════════════════════════════════════
# Phase 2: Empirical Calibration From Historical Data
# ══════════════════════════════════════════════════════════════════════════════

def calibrate_from_history(events_df: pd.DataFrame,
                           returns_df: pd.DataFrame,
                           forward_days: int = 3,
                           alpha: float = 0.6) -> pd.DataFrame:
    """
    Improve hand-crafted priors using actual event-return correlations.
    
    For each event in history, records the N-day forward return of each
    sector ETF, then computes mean impact per event type.
    
    Args:
        events_df: Must have columns [event_date, war_event_type, severity_score]
        returns_df: Sector ETF daily returns, columns = tickers
        forward_days: Look-ahead window for measuring impact
        alpha: Blending weight (alpha * empirical + (1-alpha) * prior)
        
    Returns:
        Updated sensitivity DataFrame
    """
    log_step(logger, 'Phase 2: Calibrating sensitivity from historical data',
             f'Events: {len(events_df)}, forward_days={forward_days}, alpha={alpha}')

    global SENSITIVITY_DF

    # Map event_type column name
    et_col = 'war_event_type' if 'war_event_type' in events_df.columns else 'event_type'

    rows = []
    for _, ev in events_df.iterrows():
        d = ev['event_date']
        if d not in returns_df.index:
            # Find nearest trading day
            mask = returns_df.index >= d
            if not mask.any():
                continue
            d = returns_df.index[mask][0]

        future = returns_df.loc[d:].head(forward_days + 1).iloc[1:]
        if len(future) < forward_days:
            continue

        cumret = (1 + future).prod() - 1
        rows.append({
            'event_type': ev[et_col],
            'severity': ev.get('severity_score', 1.0),
            **cumret.to_dict()
        })

    if not rows:
        logger.warning('No valid event-return pairs found for calibration')
        return SENSITIVITY_DF

    df = pd.DataFrame(rows)

    # Compute mean sector return per event type
    sector_cols = [c for c in df.columns if c in SECTORS]
    calibrated = df.groupby('event_type')[sector_cols].mean()

    # Blend with prior
    for et in calibrated.index:
        if et in SENSITIVITY_DF.index:
            shared_cols = [c for c in sector_cols if c in SENSITIVITY_DF.columns]
            SENSITIVITY_DF.loc[et, shared_cols] = (
                alpha * calibrated.loc[et, shared_cols] +
                (1 - alpha) * SENSITIVITY_DF.loc[et, shared_cols]
            )

    logger.info(f'Calibrated {len(calibrated)} event types using {len(rows)} observations')
    return SENSITIVITY_DF


# ══════════════════════════════════════════════════════════════════════════════
# Phase 3: Learned Sensitivity via Lasso Regression
# ══════════════════════════════════════════════════════════════════════════════

def learn_sensitivity_lasso(events_df: pd.DataFrame,
                            returns_df: pd.DataFrame,
                            alpha: float = LASSO_ALPHA):
    """
    Learn sparse sensitivity weights using MultiTaskLasso regression.
    
    X: One-hot encoded event types + severity + region
    y: 3-day forward returns for all 11 sectors simultaneously
    
    MultiTaskLasso enforces sparsity across all sector predictions
    simultaneously, zeroing out event types with no genuine sector impact.
    
    Args:
        events_df: Must have columns [event_date, war_event_type, country,
                   fatality_bin, severity_score]
        returns_df: Sector ETF daily returns
        alpha: L1 penalty strength
        
    Returns:
        Tuple of (fitted model, scaler, feature column names)
    """
    log_step(logger, 'Phase 3: Learning sensitivity via MultiTaskLasso',
             f'alpha={alpha}')

    et_col = 'war_event_type' if 'war_event_type' in events_df.columns else 'event_type'

    # Build feature matrix
    cat_cols = [c for c in [et_col, 'country', 'fatality_bin']
                if c in events_df.columns]
    X_enc = pd.get_dummies(events_df[cat_cols])
    X_enc['severity'] = events_df['severity_score'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_enc)

    # Build target matrix: forward returns aligned to event dates
    Y = []
    valid_idx = []
    for i, row in events_df.iterrows():
        d = row['event_date']
        if d not in returns_df.index:
            mask = returns_df.index >= d
            if not mask.any():
                continue
            d = returns_df.index[mask][0]

        fut = returns_df.loc[d:].head(4).iloc[1:]
        if len(fut) < 3:
            continue
        Y.append((1 + fut).prod().values - 1)
        valid_idx.append(i)

    if len(Y) < 10:
        logger.warning(f'Only {len(Y)} valid samples, need at least 10 for Lasso')
        return None, scaler, X_enc.columns.tolist()

    Y = np.array(Y)
    X_valid = X_scaled[events_df.index.isin(valid_idx)]

    model = MultiTaskLasso(alpha=alpha, max_iter=5000)
    model.fit(X_valid, Y)

    # model.coef_ shape: (n_sectors, n_features)
    n_nonzero = np.count_nonzero(model.coef_)
    total = model.coef_.size
    logger.info(f'Lasso fit complete: {n_nonzero}/{total} non-zero coefficients '
                f'({n_nonzero/total:.1%} density)')

    return model, scaler, X_enc.columns.tolist()


def get_sensitivity_for_event(event_type: str) -> pd.Series:
    """Get the sector impact vector for a specific event type."""
    if event_type in SENSITIVITY_DF.index:
        return SENSITIVITY_DF.loc[event_type]
    logger.warning(f'Unknown event type: {event_type}, returning zeros')
    return pd.Series(0.0, index=SECTORS)


def get_top_impacted_sectors(event_type: str, top_n: int = 5) -> pd.Series:
    """Get the top N most impacted sectors for an event type."""
    impact = get_sensitivity_for_event(event_type)
    return impact.abs().nlargest(top_n)


if __name__ == '__main__':
    print('Sensitivity Matrix (Phase 1 — Hand-Crafted Prior):')
    print(SENSITIVITY_DF.to_string())
    print(f'\nShape: {SENSITIVITY_DF.shape}')
    print(f'\nTop sectors for oil_price_spike:')
    print(get_top_impacted_sectors('oil_price_spike'))
