from __future__ import annotations
"""
Counterfactual Scenario Simulator (Module 3).

Allows users to modify oil price assumptions or event severity
and re-run the GNN to produce hypothetical sector return forecasts.

Also includes shock propagation simulation (Module 7):
Simulates how a new war event propagates sector-by-sector over
multiple time steps using graph diffusion.
"""

import copy
import numpy as np
import pandas as pd

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import SECTOR_ETFS
from utils.logger import get_logger, log_step

logger = get_logger('model.counterfactual')

SECTORS = list(SECTOR_ETFS.keys())


def run_counterfactual(model, base_snapshots: list,
                       event_embs: torch.Tensor,
                       oil_override: float = None,
                       event_suppression: list[str] = None,
                       event_amplification: dict = None) -> dict:
    """
    Run a counterfactual analysis by modifying inputs and comparing predictions.
    
    Args:
        model: Trained SectorWarGNN model
        base_snapshots: Original graph snapshots
        event_embs: Original event embeddings (T, event_emb_dim)
        oil_override: New assumed WTI price for commodity beta recompute
        event_suppression: List of event_type strings to zero out
        event_amplification: Dict of {event_type: multiplier} to amplify events
        
    Returns:
        Dict with base_preds, cf_preds, delta, and sector-level analysis
    """
    import torch
    log_step(logger, 'Running counterfactual analysis',
             f'oil_override={oil_override}, suppressed={event_suppression}')

    model.eval()

    # Clone and modify snapshots
    cf_snapshots = _deep_copy_snapshots(base_snapshots)
    cf_embs = event_embs.clone()

    # Apply oil price override
    if oil_override is not None:
        cf_snapshots = _modify_oil_features(cf_snapshots, oil_override)

    # Apply event suppression
    if event_suppression:
        cf_embs = _suppress_events(cf_embs, event_suppression)

    # Apply event amplification
    if event_amplification:
        cf_embs = _amplify_events(cf_embs, event_amplification)

    # Run both scenarios
    with torch.no_grad():
        base_preds = model(base_snapshots, event_embs).squeeze(-1).numpy()
        cf_preds = model(cf_snapshots, cf_embs).squeeze(-1).numpy()

    delta = cf_preds - base_preds  # Impact of the intervention

    # Analyze results
    result = {
        'base_preds': base_preds,
        'cf_preds': cf_preds,
        'delta': delta,
        'mean_delta': pd.Series(delta.mean(axis=0), index=SECTORS),
        'max_delta': pd.Series(delta.max(axis=0), index=SECTORS),
        'min_delta': pd.Series(delta.min(axis=0), index=SECTORS),
    }

    logger.info(f'Counterfactual delta range: [{delta.min():.4f}, {delta.max():.4f}]')
    return result


def _deep_copy_snapshots(snapshots: list) -> list:
    """Deep copy graph snapshots."""
    copied = []
    for s in snapshots:
        if isinstance(s, dict):
            copied.append({k: v.clone() for k, v in s.items()})
        else:
            copied.append(copy.deepcopy(s))
    return copied


def _modify_oil_features(snapshots: list, oil_price: float) -> list:
    """
    Modify commodity beta features in snapshots based on oil price assumption.
    Feature index 4 is commodity_beta.
    """
    for s in snapshots:
        if isinstance(s, dict):
            x = s['x']
        else:
            x = s.x

        # Scale commodity beta by oil price ratio
        # Assumes baseline of ~$80/barrel
        ratio = oil_price / 80.0
        x[:, 4] *= ratio

    return snapshots


def _suppress_events(event_embs: torch.Tensor,
                     event_types: list[str]) -> torch.Tensor:
    """Zero out specific event type signals in the embedding tensor."""
    # In practice, you'd map event types to embedding dimensions
    # For now, we scale down the overall signal
    cf_embs = event_embs.clone()
    scale = 1.0 - (len(event_types) / 15.0)  # Proportional suppression
    cf_embs *= max(scale, 0.1)
    return cf_embs


def _amplify_events(event_embs: torch.Tensor,
                    amplification: dict) -> torch.Tensor:
    """Amplify specific event type signals."""
    cf_embs = event_embs.clone()
    for event_type, multiplier in amplification.items():
        cf_embs *= multiplier
    return cf_embs


# ══════════════════════════════════════════════════════════════════════════════
# Module 7: Shock Propagation Simulation
# ══════════════════════════════════════════════════════════════════════════════

def simulate_shock_propagation(impact_vector: np.ndarray,
                                adj_matrix: np.ndarray,
                                steps: int = 5,
                                decay: float = 0.6) -> np.ndarray:
    """
    Simulate how a war event shock propagates through the sector graph.
    
    Uses a graph diffusion algorithm: starting with an initial impact vector,
    repeatedly multiply by the normalized adjacency matrix to show how
    the shock spreads sector-by-sector over time.
    
    Args:
        impact_vector: Initial sector impacts, shape (11,)
        adj_matrix: Normalized edge weight matrix, shape (11, 11)
        steps: Number of propagation time steps
        decay: Decay factor per step (0-1)
        
    Returns:
        Propagation trace, shape (steps+1, 11)
    """
    log_step(logger, 'Simulating shock propagation',
             f'steps={steps}, decay={decay}')

    trace = [impact_vector.copy()]
    current = impact_vector.copy()

    for step in range(steps):
        # Propagate through adjacency matrix
        propagated = adj_matrix @ current * decay
        current = current + propagated

        # Clip to prevent explosion
        current = np.clip(current, -5.0, 5.0)
        trace.append(current.copy())

        logger.debug(f'Step {step + 1}: max_impact={current.max():.4f}, '
                    f'affected_sectors={np.sum(np.abs(current) > 0.1)}')

    result = np.stack(trace)  # (steps+1, 11)
    logger.info(f'Propagation trace shape: {result.shape}')
    return result


def build_propagation_dataframe(trace: np.ndarray) -> pd.DataFrame:
    """
    Convert propagation trace to a labeled DataFrame for visualization.
    
    Returns:
        DataFrame with rows as time steps, columns as sectors
    """
    return pd.DataFrame(
        trace,
        columns=SECTORS,
        index=[f'Step {i}' for i in range(trace.shape[0])]
    )


def run_scenario(scenario_name: str,
                 sensitivity_df: pd.DataFrame,
                 adj_matrix: np.ndarray,
                 event_type: str = None,
                 custom_impact: np.ndarray = None,
                 steps: int = 5,
                 decay: float = 0.6) -> dict:
    """
    Run a named counterfactual scenario with shock propagation.
    
    Predefined scenarios:
    - 'hormuz_closure': Strait of Hormuz blocked
    - 'ceasefire': Broad ceasefire agreement
    - 'iran_strike': Military strike on Iran
    - 'oil_embargo': OPEC embargo on Western nations
    
    Args:
        scenario_name: Name of predefined scenario or 'custom'
        sensitivity_df: Sensitivity matrix
        adj_matrix: Adjacency matrix for propagation
        event_type: Event type for custom scenarios
        custom_impact: Custom impact vector for 'custom' scenario
        steps: Propagation steps
        decay: Decay factor
        
    Returns:
        Dict with scenario analysis results
    """
    scenarios = {
        'hormuz_closure': {
            'event_type': 'oil_route_threat',
            'severity_mult': 2.0,
            'description': 'Strait of Hormuz blocked by Iranian forces'
        },
        'ceasefire': {
            'event_type': 'ceasefire_signal',
            'severity_mult': 1.5,
            'description': 'Broad ceasefire agreement across MENA'
        },
        'iran_strike': {
            'event_type': 'us_iran_tension',
            'severity_mult': 3.0,
            'description': 'US/Israel military strike on Iranian nuclear facilities'
        },
        'oil_embargo': {
            'event_type': 'opec_cut_announcement',
            'severity_mult': 2.5,
            'description': 'OPEC+ announces severe production cuts'
        },
    }

    if scenario_name in scenarios:
        sc = scenarios[scenario_name]
        impact = sensitivity_df.loc[sc['event_type']].values * sc['severity_mult']
        desc = sc['description']
    elif event_type and event_type in sensitivity_df.index:
        impact = sensitivity_df.loc[event_type].values
        desc = f'Custom scenario: {event_type}'
    elif custom_impact is not None:
        impact = custom_impact
        desc = 'Custom impact vector'
    else:
        raise ValueError(f'Unknown scenario: {scenario_name}')

    trace = simulate_shock_propagation(impact, adj_matrix, steps, decay)
    trace_df = build_propagation_dataframe(trace)

    return {
        'scenario': scenario_name,
        'description': desc,
        'initial_impact': pd.Series(impact, index=SECTORS),
        'trace': trace,
        'trace_df': trace_df,
        'final_impact': pd.Series(trace[-1], index=SECTORS),
        'most_affected': SECTORS[np.argmax(np.abs(trace[-1]))],
        'cumulative_impact': pd.Series(trace.sum(axis=0), index=SECTORS),
    }


if __name__ == '__main__':
    from features.sensitivity_matrix import SENSITIVITY_DF
    from data.bea_io import build_io_matrix

    adj = build_io_matrix()

    for scenario in ['hormuz_closure', 'ceasefire', 'iran_strike', 'oil_embargo']:
        result = run_scenario(scenario, SENSITIVITY_DF, adj)
        print(f'\n{"=" * 60}')
        print(f'Scenario: {result["description"]}')
        print(f'Most affected sector: {result["most_affected"]}')
        print(f'Final impact:\n{result["final_impact"].round(3)}')
