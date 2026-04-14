from __future__ import annotations
"""
Main entry point for the Sector War Graph project.

Orchestrates the pipeline:
  1. Data ingestion (ACLED, GDELT, yfinance, BEA)
  2. Feature engineering (sensitivity matrix, node/edge features)
  3. Graph construction (NetworkX snapshots)
  4. Regime detection (HMM)
  5. Dashboard launch (Dash)
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from config import SECTOR_ETFS, START_DATE, END_DATE
from utils.logger import get_logger, log_step
from utils.data_store import DataStore

logger = get_logger('main')

SECTORS = list(SECTOR_ETFS.keys())


def run_pipeline(use_cache: bool = True,
                 launch_dashboard: bool = True):
    """
    Execute the pipeline from data ingestion to dashboard.

    Args:
        use_cache: Use cached data when available
        launch_dashboard: Whether to launch the Dash dashboard
    """
    logger.info('=' * 70)
    logger.info('SECTOR WAR GRAPH — Full Pipeline')
    logger.info('=' * 70)

    store = DataStore()

    # ══════════════════════════════════════════════════════════════════════
    # STEP 1: Data Ingestion
    # ══════════════════════════════════════════════════════════════════════
    log_step(logger, 'STEP 1: Data Ingestion')

    # 1a. ACLED conflict events
    from data.acled_pipeline import fetch_acled
    events_df = fetch_acled(START_DATE, END_DATE)
    logger.info(f'ACLED events: {len(events_df)} records')

    # 1b. GDELT headlines
    from data.gdelt_pipeline import fetch_gdelt_headlines
    gdelt_df = fetch_gdelt_headlines(start_date=START_DATE, end_date=END_DATE)
    logger.info(f'GDELT headlines: {len(gdelt_df)} articles')

    # 1c. Sector ETF prices
    from data.yfinance_pipeline import (fetch_sector_prices, fetch_oil_prices,
                                         compute_sector_returns, compute_monthly_returns)
    prices_df = fetch_sector_prices(START_DATE, END_DATE)
    oil_prices = fetch_oil_prices(START_DATE, END_DATE)
    returns_df = compute_sector_returns(prices_df)
    monthly_returns = compute_monthly_returns(prices_df)
    logger.info(f'Price data: {prices_df.shape[0]} trading days')

    # 1d. BEA I-O tables
    from data.bea_io import build_io_matrix, build_supply_chain_matrix
    io_matrix = build_io_matrix()
    supply_matrix = build_supply_chain_matrix()
    logger.info(f'I-O matrix: {io_matrix.shape}')

    # ══════════════════════════════════════════════════════════════════════
    # STEP 2: Sensitivity Matrix
    # ══════════════════════════════════════════════════════════════════════
    log_step(logger, 'STEP 2: Building Sensitivity Matrix')

    from features.sensitivity_matrix import (get_sensitivity_matrix,
                                              calibrate_from_history)
    sensitivity_df = get_sensitivity_matrix()
    logger.info(f'Phase 1 sensitivity matrix: {sensitivity_df.shape}')

    # Phase 2: Calibrate from historical data
    try:
        sensitivity_df = calibrate_from_history(events_df, returns_df)
        logger.info('Phase 2 calibration complete')
    except Exception as e:
        logger.warning(f'Phase 2 calibration skipped: {e}')

    # ══════════════════════════════════════════════════════════════════════
    # STEP 3: Feature Engineering
    # ══════════════════════════════════════════════════════════════════════
    log_step(logger, 'STEP 3: Feature Engineering')

    # 3a. Node features
    from features.node_features import build_node_feature_tensor
    node_features = build_node_feature_tensor(prices_df, oil_prices)
    logger.info(f'Node features: {node_features.shape}')

    # 3b. Edge features
    from features.edge_features import (rolling_correlation_matrix,
                                         oil_sensitivity_matrix,
                                         build_edge_feature_tensor,
                                         build_edge_index)
    corr_dict = rolling_correlation_matrix(returns_df)
    oil_dict = oil_sensitivity_matrix(returns_df, oil_prices)
    edge_features = build_edge_feature_tensor(corr_dict, io_matrix,
                                               oil_dict, supply_matrix)
    edge_index = build_edge_index(n_nodes=11, directed=True)
    logger.info(f'Edge features: {edge_features.shape}')

    # Align temporal dimensions
    T = min(node_features.shape[0], edge_features.shape[0])
    node_features = node_features[:T]
    edge_features = edge_features[:T]
    logger.info(f'Temporal alignment: T={T} months')

    # Build month list aligned with feature tensor
    month_dates = sorted(set(corr_dict.keys()) & set(oil_dict.keys()))[:T]
    months = [str(d.to_period('M')) for d in month_dates]

    # ══════════════════════════════════════════════════════════════════════
    # STEP 4: Graph Construction
    # ══════════════════════════════════════════════════════════════════════
    log_step(logger, 'STEP 4: Building Graph Snapshots')

    # ══════════════════════════════════════════════════════════════════════
    # STEP 6: Regime Detection
    # ══════════════════════════════════════════════════════════════════════
    log_step(logger, 'STEP 5: Regime Detection')

    from model.regime_detector import RegimeDetector, build_regime_features
    try:
        regime_features = build_regime_features(oil_prices, events_df, gdelt_df)
        detector = RegimeDetector()
        detector.fit(regime_features)
        regime_labels = detector.predict(regime_features)

        # Align to graph months
        if len(regime_labels) >= T:
            regime_labels = regime_labels[:T]
        else:
            regime_labels = np.pad(regime_labels, (0, T - len(regime_labels)),
                                   constant_values=1)

        current_regime, regime_name = detector.get_current_regime(regime_features)
        logger.info(f'Current regime: {regime_name} (label={current_regime})')
    except Exception as e:
        logger.warning(f'Regime detection failed: {e}')
        regime_labels = np.ones(T, dtype=int)

    # ══════════════════════════════════════════════════════════════════════
    # STEP 6: Visualization Data Prep
    # ══════════════════════════════════════════════════════════════════════
    log_step(logger, 'STEP 6: Preparing Visualization Data')

    from viz.graph_renderer import build_networkx_graph, graph_to_cytoscape_elements

    # Build cytoscape elements for each month
    cyto_elements = {}
    for t_idx, month in enumerate(months):
        if t_idx < T:
            nf = node_features[t_idx]
            # Use edge features at this time step for adjacency
            corr_mat = edge_features[t_idx, :, :, 0]  # Correlation
            G = build_networkx_graph(corr_mat, nf, directed=True, edge_threshold=0.1)
            cyto_elements[month] = graph_to_cytoscape_elements(G)

    # Align monthly returns
    if len(monthly_returns) > T:
        monthly_returns_aligned = monthly_returns.iloc[:T]
    else:
        monthly_returns_aligned = monthly_returns

    # ══════════════════════════════════════════════════════════════════════
    # STEP 7: Launch Dashboard
    # ══════════════════════════════════════════════════════════════════════
    data_bundle = {
        'node_features': node_features,
        'edge_features': edge_features,
        'monthly_returns': monthly_returns_aligned,
        'events_df': events_df,
        'gdelt_df': gdelt_df,
        'sensitivity_df': sensitivity_df,
        'regime_labels': regime_labels,
        'months': months,
        'cytoscape_elements': cyto_elements,
    }

    logger.info('=' * 70)
    logger.info('PIPELINE COMPLETE')
    logger.info(f'  Months: {T}')
    logger.info(f'  Events: {len(events_df)}')
    logger.info(f'  Node features: {node_features.shape}')
    logger.info(f'  Edge features: {edge_features.shape}')
    logger.info('=' * 70)

    if launch_dashboard:
        log_step(logger, 'STEP 7: Launching Dashboard')
        from viz.dashboard import run_dashboard
        run_dashboard(data_bundle)

    return data_bundle


def main():
    parser = argparse.ArgumentParser(
        description='Sector War Graph — Middle East War Impact on Sectors'
    )
    parser.add_argument('--no-dashboard', action='store_true',
                       help='Skip dashboard launch')
    parser.add_argument('--no-cache', action='store_true',
                       help='Disable data caching')

    args = parser.parse_args()

    run_pipeline(
        use_cache=not args.no_cache,
        launch_dashboard=not args.no_dashboard,
    )


if __name__ == '__main__':
    main()
