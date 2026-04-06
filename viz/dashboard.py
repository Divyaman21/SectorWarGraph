from __future__ import annotations
"""
Interactive Dash Dashboard for the Sector War Knowledge Graph.

Two-panel layout:
  LEFT:  Sector graph (dash-cytoscape) + controls
  RIGHT: Inspector panels, heatmaps, counterfactual charts

Panels:
  1. Sector graph (force-directed layout)
  2. Timeline slider (Oct 2023 → present)
  3. Edge feature selector dropdown
  4. Sector search input
  5. Node inspector (bar chart of features)
  6. Event feed (scrollable list)
  7. Rotation heatmap (11 x T z-score)
  8. Counterfactual panel (oil price slider)
  9. Regime indicator badge
  10. Attention weight chart
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import SECTOR_ETFS, REGIME_LABELS, DASH_HOST, DASH_PORT, DASH_DEBUG
from utils.logger import get_logger

logger = get_logger('viz.dashboard')

SECTORS = list(SECTOR_ETFS.keys())
SECTOR_NAMES = list(SECTOR_ETFS.values())


def create_app(data_bundle: dict = None):
    """
    Create and configure the Dash application.
    
    Args:
        data_bundle: Dict containing all precomputed data:
            - node_features: (T, 11, 5)
            - edge_features: (T, 11, 11, 4)
            - monthly_returns: DataFrame
            - events_df: DataFrame
            - gdelt_df: DataFrame
            - sensitivity_df: DataFrame
            - regime_labels: array
            - months: list of month strings
            - cytoscape_elements: list per month
            - predictions: array (optional)
            
    Returns:
        Dash app instance
    """
    import dash
    import dash_cytoscape as cyto
    from dash import dcc, html, Input, Output, State, callback_context

    cyto.load_extra_layouts()

    # ── Use provided data or generate defaults ────────────────────────────
    if data_bundle is None:
        data_bundle = _generate_default_data()

    months = data_bundle.get('months', [f'2024-{m:02d}' for m in range(1, 13)])
    n_months = len(months)

    # ══════════════════════════════════════════════════════════════════════
    # APP LAYOUT
    # ══════════════════════════════════════════════════════════════════════
    app = dash.Dash(
        __name__,
        title='Sector War Knowledge Graph',
        suppress_callback_exceptions=True
    )

    # Cytoscape stylesheet
    cyto_stylesheet = [
        {
            'selector': 'node',
            'style': {
                'label': 'data(ticker)',
                'background-color': 'data(color)',
                'width': 'data(size)',
                'height': 'data(size)',
                'font-size': '10px',
                'font-weight': 'bold',
                'text-valign': 'center',
                'text-halign': 'center',
                'border-width': 2,
                'border-color': '#333',
            }
        },
        {
            'selector': 'edge',
            'style': {
                'width': 'data(width)',
                'line-color': 'data(color)',
                'target-arrow-color': 'data(color)',
                'target-arrow-shape': 'triangle',
                'curve-style': 'bezier',
                'opacity': 0.6,
            }
        },
        {
            'selector': '.highlighted',
            'style': {
                'border-color': '#FFD700',
                'border-width': 4,
                'z-index': 999,
            }
        },
    ]

    app.layout = html.Div([
        # ── Header ───────────────────────────────────────────────────────
        html.Div([
            html.H1('🌍 Middle East War → Sector Impact Graph',
                     style={'margin': '0', 'color': '#2C3E50',
                            'fontSize': '24px'}),
            html.P('Temporal GNN-based analysis of geopolitical event propagation',
                    style={'margin': '5px 0', 'color': '#7F8C8D',
                           'fontSize': '13px'}),
        ], style={'padding': '15px 25px', 'backgroundColor': '#ECF0F1',
                  'borderBottom': '3px solid #3498DB'}),

        # ── Main Content ─────────────────────────────────────────────────
        html.Div([
            # ── LEFT PANEL: Graph + Controls ─────────────────────────────
            html.Div([
                # Controls bar
                html.Div([
                    html.Div([
                        html.Label('Search Sector', style={'fontWeight': 'bold',
                                                            'fontSize': '12px'}),
                        dcc.Input(
                            id='search-input',
                            placeholder='Type sector name...',
                            type='text',
                            style={'width': '100%', 'padding': '6px',
                                   'borderRadius': '4px', 'border': '1px solid #BDC3C7'}
                        ),
                    ], style={'flex': '1', 'marginRight': '10px'}),

                    html.Div([
                        html.Label('Edge Feature', style={'fontWeight': 'bold',
                                                           'fontSize': '12px'}),
                        dcc.Dropdown(
                            id='edge-mode',
                            options=[
                                {'label': '📊 Correlation', 'value': 'corr'},
                                {'label': '🏭 Input-Output', 'value': 'io'},
                                {'label': '🛢️ Oil Sensitivity', 'value': 'oil'},
                                {'label': '🔗 Supply Chain', 'value': 'supply'},
                            ],
                            value='corr',
                            clearable=False,
                            style={'fontSize': '13px'}
                        ),
                    ], style={'flex': '1'}),
                ], style={'display': 'flex', 'marginBottom': '10px'}),

                # Cytoscape graph
                cyto.Cytoscape(
                    id='sector-graph',
                    layout={'name': 'cose', 'animate': True,
                            'nodeRepulsion': 8000, 'idealEdgeLength': 100},
                    style={'width': '100%', 'height': '420px',
                           'backgroundColor': '#FAFAFA',
                           'border': '1px solid #ECF0F1',
                           'borderRadius': '8px'},
                    elements=data_bundle.get('cytoscape_elements', {}).get(
                        months[-1] if months else '2024-01', []),
                    stylesheet=cyto_stylesheet,
                ),

                # Timeline slider
                html.Div([
                    html.Label('📅 Timeline', style={'fontWeight': 'bold',
                                                      'fontSize': '12px',
                                                      'marginBottom': '5px'}),
                    dcc.Slider(
                        id='timeline-slider',
                        min=0,
                        max=max(n_months - 1, 1),
                        value=max(n_months - 1, 0),
                        marks={i: {'label': months[i][-7:] if i < n_months else '',
                                   'style': {'fontSize': '10px'}}
                               for i in range(0, n_months, max(n_months // 6, 1))},
                        tooltip={'placement': 'bottom'},
                    ),
                ], style={'marginTop': '10px', 'padding': '0 10px'}),

            ], style={
                'width': '54%',
                'display': 'inline-block',
                'verticalAlign': 'top',
                'padding': '10px 15px',
            }),

            # ── RIGHT PANEL: Inspector + Analytics ─────────────────────
            html.Div([
                # Regime badge
                html.Div([
                    html.Span('Current Regime: ', style={'fontWeight': 'bold',
                                                          'fontSize': '13px'}),
                    html.Span(id='regime-badge',
                             children='Plateau',
                             style={
                                 'padding': '4px 12px',
                                 'borderRadius': '12px',
                                 'backgroundColor': '#F39C12',
                                 'color': 'white',
                                 'fontWeight': 'bold',
                                 'fontSize': '12px',
                             }),
                ], style={'marginBottom': '12px', 'textAlign': 'center'}),

                # Node inspector
                html.Div([
                    html.H4('🔍 Node Inspector', style={'margin': '0 0 5px 0',
                                                          'fontSize': '14px'}),
                    dcc.Graph(id='node-inspector',
                             style={'height': '200px'},
                             config={'displayModeBar': False}),
                ], style={'marginBottom': '12px',
                          'border': '1px solid #ECF0F1',
                          'borderRadius': '8px', 'padding': '10px'}),

                # Rotation heatmap
                html.Div([
                    html.H4('🗺️ Sector Rotation Heatmap', style={'margin': '0 0 5px 0',
                                                                    'fontSize': '14px'}),
                    dcc.Graph(id='rotation-heatmap',
                             style={'height': '250px'},
                             config={'displayModeBar': False}),
                ], style={'marginBottom': '12px',
                          'border': '1px solid #ECF0F1',
                          'borderRadius': '8px', 'padding': '10px'}),

                # Counterfactual panel
                html.Div([
                    html.H4('🔮 What-If Scenario', style={'margin': '0 0 5px 0',
                                                            'fontSize': '14px'}),
                    html.Div([
                        html.Label('Oil Price ($/bbl):', style={'fontSize': '12px'}),
                        dcc.Slider(
                            id='oil-price-slider',
                            min=40, max=200, value=80, step=5,
                            marks={40: '$40', 80: '$80', 120: '$120',
                                   160: '$160', 200: '$200'},
                            tooltip={'placement': 'bottom'},
                        ),
                    ]),
                    dcc.Graph(id='counterfactual-chart',
                             style={'height': '180px'},
                             config={'displayModeBar': False}),
                ], style={'marginBottom': '12px',
                          'border': '1px solid #ECF0F1',
                          'borderRadius': '8px', 'padding': '10px'}),

                # Event feed
                html.Div([
                    html.H4('📰 Recent Events', style={'margin': '0 0 5px 0',
                                                         'fontSize': '14px'}),
                    html.Div(id='event-feed',
                             style={'maxHeight': '200px', 'overflowY': 'auto',
                                    'fontSize': '12px'}),
                ], style={'border': '1px solid #ECF0F1',
                          'borderRadius': '8px', 'padding': '10px'}),

            ], style={
                'width': '44%',
                'display': 'inline-block',
                'verticalAlign': 'top',
                'padding': '10px 15px',
            }),
        ]),
    ], style={'fontFamily': 'Segoe UI, Roboto, sans-serif',
              'backgroundColor': '#FFFFFF'})

    # ══════════════════════════════════════════════════════════════════════
    # CALLBACKS
    # ══════════════════════════════════════════════════════════════════════

    @app.callback(
        Output('sector-graph', 'elements'),
        [Input('timeline-slider', 'value'),
         Input('edge-mode', 'value'),
         Input('search-input', 'value')],
    )
    def update_graph(timeline_idx, edge_mode, search_text):
        """Update graph elements based on timeline, edge mode, and search."""
        if timeline_idx is None or timeline_idx >= len(months):
            return []

        month = months[timeline_idx]
        elements = data_bundle.get('cytoscape_elements', {}).get(month, [])

        # Apply search filter
        if search_text:
            search_lower = search_text.lower()
            for elem in elements:
                if 'source' not in elem.get('data', {}):  # Node
                    name = elem['data'].get('name', '').lower()
                    ticker = elem['data'].get('ticker', '').lower()
                    if search_lower in name or search_lower in ticker:
                        elem['classes'] = 'highlighted'
                    else:
                        elem['classes'] = ''

        return elements

    @app.callback(
        Output('regime-badge', 'children'),
        Output('regime-badge', 'style'),
        Input('timeline-slider', 'value'),
    )
    def update_regime_badge(timeline_idx):
        """Update the regime indicator badge."""
        regime_labels = data_bundle.get('regime_labels', np.array([1]))

        if timeline_idx is not None and timeline_idx < len(regime_labels):
            regime = int(regime_labels[timeline_idx])
        else:
            regime = 1

        label = REGIME_LABELS.get(regime, 'Unknown')
        colors = {0: '#E74C3C', 1: '#F39C12', 2: '#2ECC71'}
        color = colors.get(regime, '#95A5A6')

        style = {
            'padding': '4px 12px',
            'borderRadius': '12px',
            'backgroundColor': color,
            'color': 'white',
            'fontWeight': 'bold',
            'fontSize': '12px',
        }

        return label, style

    @app.callback(
        Output('node-inspector', 'figure'),
        Input('sector-graph', 'tapNodeData'),
    )
    def update_node_inspector(node_data):
        """Show feature bar chart for selected sector node."""
        if node_data is None:
            # Default: show all sectors' volatility
            fig = go.Figure(go.Bar(
                x=SECTORS,
                y=[0.2] * 11,
                marker_color=[_get_color(s) for s in SECTORS],
            ))
            fig.update_layout(
                title='Click a node to inspect',
                margin=dict(l=30, r=10, t=30, b=30),
                height=180,
                plot_bgcolor='white',
            )
            return fig

        features = {
            'Returns': node_data.get('returns', 0),
            'Volatility': node_data.get('volatility', 0),
            'Momentum': node_data.get('momentum', 0),
            'Valuation': node_data.get('valuation', 0),
            'Oil Beta': node_data.get('comm_beta', 0),
        }

        ticker = node_data.get('ticker', '?')
        name = node_data.get('name', '?')

        fig = go.Figure(go.Bar(
            x=list(features.keys()),
            y=list(features.values()),
            marker_color=['#3498DB', '#E74C3C', '#2ECC71', '#F39C12', '#9B59B6'],
        ))

        fig.update_layout(
            title=f'{name} ({ticker})',
            margin=dict(l=30, r=10, t=30, b=30),
            height=180,
            plot_bgcolor='white',
            yaxis_title='Value',
        )
        return fig

    @app.callback(
        Output('rotation-heatmap', 'figure'),
        Input('timeline-slider', 'value'),
    )
    def update_heatmap(timeline_idx):
        """Update the sector rotation heatmap."""
        monthly_returns = data_bundle.get('monthly_returns')
        if monthly_returns is None:
            return go.Figure()

        from viz.heatmap import build_rotation_heatmap
        regime_labels = data_bundle.get('regime_labels')
        events_df = data_bundle.get('events_df')

        fig = build_rotation_heatmap(monthly_returns, events_df, regime_labels)
        fig.update_layout(height=230, margin=dict(l=120, r=10, t=30, b=30))
        return fig

    @app.callback(
        Output('counterfactual-chart', 'figure'),
        Input('oil-price-slider', 'value'),
    )
    def update_counterfactual(oil_price):
        """Update counterfactual chart based on oil price assumption."""
        from features.sensitivity_matrix import SENSITIVITY_DF

        # Simple impact estimation based on oil price
        baseline = 80.0
        ratio = (oil_price - baseline) / baseline

        # Scale oil-related event impacts
        oil_events = ['oil_route_threat', 'oil_price_spike', 'opec_cut_announcement']
        impact = pd.Series(0.0, index=SECTORS)
        for et in oil_events:
            if et in SENSITIVITY_DF.index:
                impact += SENSITIVITY_DF.loc[et] * ratio

        colors = ['#E74C3C' if v > 0 else '#2ECC71' for v in impact.values]

        fig = go.Figure(go.Bar(
            x=[SECTOR_ETFS.get(s, s) for s in impact.index],
            y=impact.values,
            marker_color=colors,
        ))

        fig.update_layout(
            title=f'Oil @ ${oil_price}/bbl (Δ from $80)',
            margin=dict(l=30, r=10, t=30, b=60),
            height=160,
            plot_bgcolor='white',
            yaxis_title='Impact',
            xaxis_tickangle=-45,
        )
        return fig

    @app.callback(
        Output('event-feed', 'children'),
        Input('timeline-slider', 'value'),
    )
    def update_event_feed(timeline_idx):
        """Update the event feed for the selected time period."""
        events_df = data_bundle.get('events_df')
        if events_df is None or len(events_df) == 0:
            return html.P('No events loaded', style={'color': '#95A5A6'})

        if timeline_idx is not None and timeline_idx < len(months):
            month = months[timeline_idx]
            # Filter events up to this month
            mask = events_df['event_date'] <= pd.Timestamp(month + '-28')
            recent = events_df[mask].tail(20)
        else:
            recent = events_df.tail(20)

        items = []
        for _, ev in recent.iterrows():
            severity = ev.get('severity_score', 1.0)
            if severity > 5:
                emoji = '🔴'
            elif severity > 3:
                emoji = '🟠'
            else:
                emoji = '🟡'

            date_str = ev['event_date'].strftime('%Y-%m-%d')
            event_type = ev.get('war_event_type', ev.get('event_type', 'unknown'))
            country = ev.get('country', 'Unknown')
            text = ev.get('notes', ev.get('title', ''))[:80]

            items.append(html.Div([
                html.Span(f'{emoji} ', style={'fontSize': '14px'}),
                html.Strong(f'[{date_str}] ', style={'color': '#2C3E50'}),
                html.Span(f'{event_type} ', style={'color': '#3498DB'}),
                html.Span(f'({country}) ', style={'color': '#7F8C8D'}),
                html.Br(),
                html.Span(text, style={'color': '#555', 'fontSize': '11px'}),
            ], style={'padding': '4px 0', 'borderBottom': '1px solid #ECF0F1'}))

        return items

    return app


def _get_color(sector: str) -> str:
    """Get color for a sector."""
    from viz.graph_renderer import SECTOR_COLORS
    return SECTOR_COLORS.get(sector, '#888')


def _generate_default_data() -> dict:
    """Generate default/demo data for the dashboard."""
    logger.info('Generating default dashboard data')
    np.random.seed(42)

    months = [f'2024-{m:02d}' for m in range(1, 13)]
    n = len(months)

    # Synthetic monthly returns
    returns_data = np.random.randn(n, 11) * 0.05
    monthly_returns = pd.DataFrame(
        returns_data, index=pd.to_datetime(months),
        columns=SECTORS
    )

    # Synthetic events
    from data.acled_pipeline import _generate_synthetic_acled, _classify_events, _compute_severity
    events_df = _generate_synthetic_acled('2024-01-01', '2024-12-31')
    events_df = _classify_events(events_df)
    events_df = _compute_severity(events_df)

    # Regime labels
    regime_labels = np.random.choice([0, 1, 2], n, p=[0.3, 0.4, 0.3])

    # Cytoscape elements per month
    from data.bea_io import build_io_matrix
    from viz.graph_renderer import build_networkx_graph, graph_to_cytoscape_elements

    io = build_io_matrix()
    cyto_elements = {}
    for i, month in enumerate(months):
        node_feats = np.random.rand(11, 5) * 0.5
        G = build_networkx_graph(io, node_feats)
        cyto_elements[month] = graph_to_cytoscape_elements(G)

    return {
        'months': months,
        'monthly_returns': monthly_returns,
        'events_df': events_df,
        'regime_labels': regime_labels,
        'cytoscape_elements': cyto_elements,
    }


def run_dashboard(data_bundle: dict = None):
    """Start the Dash application."""
    app = create_app(data_bundle)
    logger.info(f'Starting dashboard at http://{DASH_HOST}:{DASH_PORT}')
    app.run(host=DASH_HOST, port=DASH_PORT, debug=DASH_DEBUG)


if __name__ == '__main__':
    run_dashboard()
