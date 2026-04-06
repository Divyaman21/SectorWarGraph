from __future__ import annotations
"""
Sector Rotation Heatmap (Module 8).

Renders an (11 sectors x T months) heatmap using Plotly.
Color = z-score of monthly return relative to cross-sector mean.
Red markers at event dates show regime transitions.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import SECTOR_ETFS
from utils.logger import get_logger

logger = get_logger('viz.heatmap')


def build_rotation_heatmap(returns_monthly: pd.DataFrame,
                            events_df: pd.DataFrame = None,
                            regime_labels: np.ndarray = None) -> go.Figure:
    """
    Build a sector rotation heatmap showing z-scored monthly returns.
    
    Args:
        returns_monthly: Monthly returns DataFrame (T x 11)
        events_df: Optional ACLED events to overlay as markers
        regime_labels: Optional HMM regime labels per month
        
    Returns:
        Plotly Figure
    """
    z = returns_monthly.T.values  # (11, T)

    # Z-score normalization across sectors per month
    z_score = (z - z.mean(axis=0, keepdims=True)) / (z.std(axis=0, keepdims=True) + 1e-8)

    # Sector names
    sector_names = [SECTOR_ETFS.get(c, c) for c in returns_monthly.columns]
    month_labels = returns_monthly.index.astype(str).tolist()

    fig = go.Figure()

    # Main heatmap
    fig.add_trace(go.Heatmap(
        z=z_score,
        x=month_labels,
        y=sector_names,
        colorscale='RdYlGn',
        zmid=0, zmin=-3, zmax=3,
        colorbar=dict(
            title='Z-Score',
            thickness=15,
            len=0.8
        ),
        hovertemplate=(
            'Sector: %{y}<br>'
            'Month: %{x}<br>'
            'Z-Score: %{z:.2f}<br>'
            '<extra></extra>'
        )
    ))

    # Overlay event markers
    if events_df is not None and 'event_date' in events_df.columns:
        events_df = events_df.copy()
        events_df['month'] = events_df['event_date'].dt.to_period('M').astype(str)
        major = events_df[events_df.get('severity_score', pd.Series(1.0)) > 3.0]

        if len(major) > 0:
            for month_str in major['month'].unique():
                if month_str in month_labels:
                    fig.add_vline(
                        x=month_str,
                        line_color='red',
                        line_width=1,
                        opacity=0.5,
                        annotation_text='⚡',
                        annotation_position='top'
                    )

    # Regime band annotations
    if regime_labels is not None:
        regime_colors = {0: 'rgba(255,0,0,0.1)',   # Escalation
                         1: 'rgba(255,255,0,0.1)',  # Plateau
                         2: 'rgba(0,255,0,0.1)'}    # De-escalation
        regime_names = {0: 'Escalation', 1: 'Plateau', 2: 'De-escalation'}

        for i, label in enumerate(regime_labels):
            if i < len(month_labels):
                fig.add_vrect(
                    x0=month_labels[max(0, i - 1)] if i > 0 else month_labels[0],
                    x1=month_labels[min(i, len(month_labels) - 1)],
                    fillcolor=regime_colors.get(label, 'rgba(128,128,128,0.1)'),
                    layer='below',
                    line_width=0,
                )

    fig.update_layout(
        title=dict(
            text='Sector Rotation Heatmap — Z-Scored Monthly Returns',
            font=dict(size=16)
        ),
        xaxis_title='Month',
        yaxis_title='Sector',
        height=500,
        margin=dict(l=150, r=50, t=80, b=50),
        plot_bgcolor='white',
        paper_bgcolor='white',
    )

    return fig


def build_propagation_heatmap(trace_df: pd.DataFrame,
                               scenario_name: str = '') -> go.Figure:
    """
    Build a heatmap showing shock propagation across sectors over time steps.
    
    Args:
        trace_df: Propagation trace DataFrame (steps x sectors)
        scenario_name: Name of the scenario for title
        
    Returns:
        Plotly Figure
    """
    sector_names = [SECTOR_ETFS.get(c, c) for c in trace_df.columns]

    fig = go.Figure(go.Heatmap(
        z=trace_df.T.values,
        x=trace_df.index.tolist(),
        y=sector_names,
        colorscale='RdBu_r',
        zmid=0,
        colorbar=dict(title='Impact', thickness=15),
        hovertemplate=(
            'Sector: %{y}<br>'
            'Step: %{x}<br>'
            'Impact: %{z:.3f}<br>'
            '<extra></extra>'
        )
    ))

    title = f'Shock Propagation — {scenario_name}' if scenario_name \
        else 'Shock Propagation Through Sector Graph'

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis_title='Propagation Step',
        yaxis_title='Sector',
        height=400,
        margin=dict(l=150, r=50, t=80, b=50),
        plot_bgcolor='white',
    )

    return fig


def build_sensitivity_heatmap(sensitivity_df: pd.DataFrame) -> go.Figure:
    """
    Build a heatmap of the event-sector sensitivity matrix.
    
    Args:
        sensitivity_df: Sensitivity matrix (events x sectors)
        
    Returns:
        Plotly Figure
    """
    sector_names = [SECTOR_ETFS.get(c, c) for c in sensitivity_df.columns]
    event_names = [et.replace('_', ' ').title() for et in sensitivity_df.index]

    fig = go.Figure(go.Heatmap(
        z=sensitivity_df.values,
        x=sector_names,
        y=event_names,
        colorscale='RdBu_r',
        zmid=0,
        colorbar=dict(title='Sensitivity', thickness=15),
        hovertemplate=(
            'Event: %{y}<br>'
            'Sector: %{x}<br>'
            'Sensitivity: %{z:.2f}<br>'
            '<extra></extra>'
        )
    ))

    fig.update_layout(
        title=dict(
            text='Event-Sector Sensitivity Matrix',
            font=dict(size=16)
        ),
        xaxis_title='Sector',
        yaxis_title='Event Type',
        height=600,
        margin=dict(l=200, r=50, t=80, b=100),
        plot_bgcolor='white',
    )

    return fig


if __name__ == '__main__':
    # Test with synthetic data
    np.random.seed(42)
    months = pd.date_range('2023-10', periods=18, freq='ME')
    data = np.random.randn(18, 11) * 0.05
    returns = pd.DataFrame(data, index=months, columns=list(SECTOR_ETFS.keys()))

    fig = build_rotation_heatmap(returns)
    fig.show()
