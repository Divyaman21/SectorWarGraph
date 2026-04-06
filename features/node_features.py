from __future__ import annotations
"""
Node Features module.
Computes all node feature vectors per sector per time step.

Features:
  1. returns       — Rolling 30d mean daily return (z-score)
  2. volatility    — Rolling 30d std * sqrt(252) (min-max 0-1)
  3. momentum      — Price[t] / Price[t-window] - 1 (z-score)
  4. valuation     — Price / 252d rolling mean (min-max 0-1)
  5. commodity_beta — Rolling 30d corr(sector returns, WTI returns) (raw -1 to 1)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import SECTOR_ETFS, WINDOW_DAYS
from utils.logger import get_logger, log_step

logger = get_logger('features.node_features')


def build_node_feature_tensor(prices_df: pd.DataFrame,
                               oil_series: pd.Series,
                               window: int = WINDOW_DAYS) -> np.ndarray:
    """
    Build node feature tensor of shape (T_months, N_sectors, 5).
    
    Each monthly snapshot contains 5 features per sector node:
    [returns, volatility, momentum, valuation, commodity_beta]
    
    Args:
        prices_df: Daily closing prices, columns = sector tickers
        oil_series: Daily oil price or returns series
        window: Rolling window in trading days
        
    Returns:
        numpy array of shape (T, 11, 5)
    """
    log_step(logger, 'Building node feature tensor',
             f'Window={window}d, sectors={prices_df.shape[1]}')

    rets = prices_df.pct_change()
    oil_rets = oil_series.pct_change() if oil_series.iloc[0] > 1 else oil_series

    # Align oil returns with sector returns
    oil_rets = oil_rets.reindex(rets.index).ffill().bfill()

    T_list = []
    month_dates = []

    for date, _ in prices_df.resample('ME').last().iterrows():
        window_data = prices_df.loc[:date].tail(window)
        r_window = rets.loc[:date].tail(window)
        oil_window = oil_rets.loc[:date].tail(window)

        if len(r_window) < window // 2:
            continue

        feat_matrix = []
        for col in prices_df.columns:
            r = r_window[col].dropna()
            p = window_data[col].dropna()

            if len(r) < 5 or len(p) < 5:
                feat_matrix.append([0.0, 0.0, 0.0, 1.0, 0.0])
                continue

            # 1. Returns: rolling mean
            returns_val = r.mean()

            # 2. Volatility: annualized
            vol_val = r.std() * np.sqrt(252)

            # 3. Momentum: price change over window
            mom_val = p.iloc[-1] / p.iloc[0] - 1 if p.iloc[0] != 0 else 0.0

            # 4. Valuation proxy: P / MA(252)
            ma_252 = prices_df[col].rolling(252, min_periods=50).mean()
            ma_val = ma_252.loc[:date].iloc[-1] if len(ma_252.loc[:date]) > 0 else p.iloc[-1]
            val_ratio = p.iloc[-1] / ma_val if ma_val != 0 else 1.0

            # 5. Commodity beta: correlation with oil
            oil_w = oil_window.reindex(r.index).dropna()
            r_aligned = r.reindex(oil_w.index).dropna()
            if len(r_aligned) > 5 and len(oil_w) > 5:
                common_idx = r_aligned.index.intersection(oil_w.index)
                comm_beta = r_aligned.loc[common_idx].corr(oil_w.loc[common_idx])
            else:
                comm_beta = 0.0

            if np.isnan(comm_beta):
                comm_beta = 0.0

            feat_matrix.append([returns_val, vol_val, mom_val, val_ratio, comm_beta])

        feat_arr = np.array(feat_matrix)  # (N_sectors, 5)
        T_list.append(feat_arr)
        month_dates.append(date)

    if not T_list:
        logger.warning('No valid monthly snapshots found')
        return np.zeros((1, prices_df.shape[1], 5))

    tensor = np.stack(T_list)  # (T, N_sectors, 5)

    # ── Normalize features ────────────────────────────────────────────────
    tensor = _normalize_features(tensor)

    logger.info(f'Node feature tensor shape: {tensor.shape}')
    return tensor


def _normalize_features(tensor: np.ndarray) -> np.ndarray:
    """
    Normalize features across sectors per time step.
    
    Feature normalization strategy:
    - returns:   z-score across sectors
    - volatility: min-max 0-1
    - momentum:  z-score
    - valuation: min-max 0-1
    - comm_beta: raw (-1 to 1)
    """
    T, N, F = tensor.shape

    for t in range(T):
        for f_idx, method in enumerate(['zscore', 'minmax', 'zscore', 'minmax', 'raw']):
            vals = tensor[t, :, f_idx]

            if method == 'zscore':
                mean = vals.mean()
                std = vals.std()
                if std > 1e-8:
                    tensor[t, :, f_idx] = (vals - mean) / std
                else:
                    tensor[t, :, f_idx] = 0.0

            elif method == 'minmax':
                vmin, vmax = vals.min(), vals.max()
                if vmax - vmin > 1e-8:
                    tensor[t, :, f_idx] = (vals - vmin) / (vmax - vmin)
                else:
                    tensor[t, :, f_idx] = 0.5

            # 'raw' — keep as-is

    return tensor


def build_node_features_single(prices_df: pd.DataFrame,
                                oil_series: pd.Series,
                                date: pd.Timestamp,
                                window: int = WINDOW_DAYS) -> np.ndarray:
    """
    Build node features for a single date snapshot.
    
    Returns:
        numpy array of shape (N_sectors, 5)
    """
    rets = prices_df.pct_change()
    oil_rets = oil_series.pct_change() if oil_series.iloc[0] > 1 else oil_series
    oil_rets = oil_rets.reindex(rets.index).ffill().bfill()

    r_window = rets.loc[:date].tail(window)
    window_data = prices_df.loc[:date].tail(window)
    oil_window = oil_rets.loc[:date].tail(window)

    feat_matrix = []
    for col in prices_df.columns:
        r = r_window[col].dropna()
        p = window_data[col].dropna()

        if len(r) < 5:
            feat_matrix.append([0.0, 0.0, 0.0, 1.0, 0.0])
            continue

        returns_val = r.mean()
        vol_val = r.std() * np.sqrt(252)
        mom_val = p.iloc[-1] / p.iloc[0] - 1 if len(p) > 0 and p.iloc[0] != 0 else 0.0

        ma_252 = prices_df[col].rolling(252, min_periods=50).mean()
        ma_val = ma_252.loc[:date].iloc[-1] if len(ma_252.loc[:date]) > 0 else p.iloc[-1]
        val_ratio = p.iloc[-1] / ma_val if ma_val != 0 else 1.0

        oil_w = oil_window.reindex(r.index).dropna()
        r_aligned = r.reindex(oil_w.index).dropna()
        if len(r_aligned) > 5:
            common_idx = r_aligned.index.intersection(oil_w.index)
            comm_beta = r_aligned.loc[common_idx].corr(oil_w.loc[common_idx])
        else:
            comm_beta = 0.0

        feat_matrix.append([returns_val, vol_val, mom_val, val_ratio,
                           comm_beta if not np.isnan(comm_beta) else 0.0])

    return np.array(feat_matrix)


if __name__ == '__main__':
    from data.yfinance_pipeline import fetch_sector_prices, fetch_oil_prices
    prices = fetch_sector_prices()
    oil = fetch_oil_prices()
    tensor = build_node_feature_tensor(prices, oil)
    print(f'Node features tensor: {tensor.shape}')
    print(f'Sample (last month):\n{tensor[-1]}')
