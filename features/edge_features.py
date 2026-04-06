from __future__ import annotations
"""
Edge Features module.
Computes all edge feature matrices between sector nodes.

Four edge features:
  1. Rolling correlation   — Undirected, dynamic (T x 11 x 11)
  2. I-O dependence        — Directed, static (11 x 11)
  3. Oil sensitivity       — Undirected, dynamic (T x 11 x 11)
  4. Supply-chain linkage  — Directed, semi-static (11 x 11)
"""

import pandas as pd
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import WINDOW_DAYS
from utils.logger import get_logger, log_step

logger = get_logger('features.edge_features')


def rolling_correlation_matrix(returns_df: pd.DataFrame,
                                window: int = WINDOW_DAYS) -> dict:
    """
    Compute rolling pairwise correlation matrices between all sector ETFs.
    
    Args:
        returns_df: Daily sector returns DataFrame
        window: Rolling window in trading days
        
    Returns:
        Dict of {date: 11x11 correlation matrix}
    """
    log_step(logger, 'Computing rolling correlation matrices',
             f'Window={window}d')

    result = {}
    for date, _ in returns_df.resample('ME').last().iterrows():
        w = returns_df.loc[:date].tail(window)
        if len(w) < window // 2:
            continue
        corr = w.corr().values
        # Replace NaN with 0
        corr = np.nan_to_num(corr, nan=0.0)
        result[date] = corr

    logger.info(f'Correlation matrices: {len(result)} monthly snapshots')
    return result


def oil_sensitivity_matrix(returns_df: pd.DataFrame,
                           oil_series: pd.Series,
                           window: int = WINDOW_DAYS) -> dict:
    """
    Compute oil co-exposure matrices.
    Cell (i,j) = beta_i * beta_j where beta = corr(sector, oil).
    
    This captures pairs of sectors that are jointly exposed to oil price risk.
    
    Args:
        returns_df: Daily sector returns
        oil_series: Daily oil returns
        window: Rolling window
        
    Returns:
        Dict of {date: 11x11 oil co-exposure matrix}
    """
    log_step(logger, 'Computing oil sensitivity matrices',
             f'Window={window}d')

    oil_rets = oil_series.pct_change() if oil_series.iloc[0] > 1 else oil_series
    oil_rets = oil_rets.reindex(returns_df.index).ffill().bfill()

    result = {}
    for date, _ in returns_df.resample('ME').last().iterrows():
        w = returns_df.loc[:date].tail(window)
        o = oil_rets.loc[:date].tail(window)

        if len(w) < window // 2:
            continue

        # Compute per-sector oil betas
        betas = w.apply(lambda s: s.corr(o.reindex(s.index)))
        betas = betas.fillna(0).values  # (11,)

        # Outer product: co-exposure matrix
        mat = np.outer(betas, betas)  # (11, 11)
        result[date] = mat

    logger.info(f'Oil sensitivity matrices: {len(result)} snapshots')
    return result


def sentiment_comovement_matrix(sentiment_scores: dict,
                                 window_weeks: int = 4) -> dict:
    """
    Module 4 improvement: Sentiment co-movement edge.
    Computes Pearson correlation of weekly FinBERT sentiment scores
    across sector-relevant news.
    
    Args:
        sentiment_scores: Dict of {date: Series indexed by sector tickers}
        window_weeks: Rolling window in weeks
        
    Returns:
        Dict of {date: 11x11 sentiment correlation matrix}
    """
    if not sentiment_scores:
        return {}

    # Convert to DataFrame
    sent_df = pd.DataFrame(sentiment_scores).T
    sent_df.index = pd.to_datetime(sent_df.index)
    sent_df = sent_df.sort_index()

    # Resample weekly
    weekly = sent_df.resample('W').mean()

    result = {}
    for i in range(window_weeks, len(weekly)):
        w = weekly.iloc[i - window_weeks:i]
        if len(w) < window_weeks // 2:
            continue
        corr = w.corr().values
        corr = np.nan_to_num(corr, nan=0.0)
        result[weekly.index[i]] = corr

    logger.info(f'Sentiment co-movement matrices: {len(result)} snapshots')
    return result


def build_edge_feature_tensor(corr_dict: dict,
                               io_matrix: np.ndarray,
                               oil_dict: dict,
                               supply_matrix: np.ndarray) -> np.ndarray:
    """
    Stack all 4 edge features into a single tensor.
    
    Args:
        corr_dict: Rolling correlation matrices {date: (11,11)}
        io_matrix: Static I-O dependency matrix (11,11)
        oil_dict: Oil co-exposure matrices {date: (11,11)}
        supply_matrix: Static supply-chain linkage matrix (11,11)
        
    Returns:
        numpy array of shape (T, 11, 11, 4)
    """
    log_step(logger, 'Building edge feature tensor')

    dates = sorted(set(corr_dict.keys()) & set(oil_dict.keys()))

    if not dates:
        logger.warning('No overlapping dates between correlation and oil matrices')
        N = io_matrix.shape[0]
        return np.zeros((1, N, N, 4))

    tensors = []
    for d in dates:
        e = np.stack([
            corr_dict[d],     # (11, 11) — dynamic correlation
            io_matrix,        # (11, 11) — static I-O
            oil_dict[d],      # (11, 11) — dynamic oil co-exposure
            supply_matrix,    # (11, 11) — semi-static supply chain
        ], axis=-1)           # (11, 11, 4)
        tensors.append(e)

    result = np.stack(tensors)  # (T, 11, 11, 4)
    logger.info(f'Edge feature tensor shape: {result.shape}')
    return result


def build_edge_index(n_nodes: int = 11,
                     directed: bool = True) -> np.ndarray:
    """
    Build a fully-connected edge index for the sector graph.
    
    Args:
        n_nodes: Number of sector nodes
        directed: If True, include both directions for each edge
        
    Returns:
        numpy array of shape (2, n_edges)
    """
    edges = []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                edges.append([i, j])
                if not directed:
                    break  # Only add (i,j), not (j,i) separately

    edge_index = np.array(edges).T  # (2, n_edges)
    logger.info(f'Edge index: {edge_index.shape[1]} edges '
                f'({"directed" if directed else "undirected"})')
    return edge_index


def extract_edge_attrs(edge_tensor_t: np.ndarray,
                       edge_index: np.ndarray) -> np.ndarray:
    """
    Extract edge attributes for a single time step from the full tensor.
    
    Args:
        edge_tensor_t: Edge features for time t, shape (11, 11, 4)
        edge_index: Edge index, shape (2, n_edges)
        
    Returns:
        numpy array of shape (n_edges, 4)
    """
    src, dst = edge_index[0], edge_index[1]
    return edge_tensor_t[src, dst]  # (n_edges, 4)


if __name__ == '__main__':
    from data.yfinance_pipeline import fetch_sector_prices, fetch_oil_prices
    from data.bea_io import build_io_matrix, build_supply_chain_matrix

    prices = fetch_sector_prices()
    oil = fetch_oil_prices()
    returns = prices.pct_change().dropna()

    corr = rolling_correlation_matrix(returns)
    oil_mat = oil_sensitivity_matrix(returns, oil)
    io = build_io_matrix()
    supply = build_supply_chain_matrix()

    tensor = build_edge_feature_tensor(corr, io, oil_mat, supply)
    print(f'Edge feature tensor: {tensor.shape}')
