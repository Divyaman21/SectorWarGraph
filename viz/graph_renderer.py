from __future__ import annotations
"""
Graph Renderer — Plotly/Networkx graph layout.

Renders the sector knowledge graph as:
1. Plotly-based interactive graph (for Dash embedding)
2. Cytoscape-compatible element format (for dash-cytoscape)
"""

import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import SECTOR_ETFS
from utils.logger import get_logger

logger = get_logger('viz.graph_renderer')

SECTORS = list(SECTOR_ETFS.keys())
SECTOR_NAMES = list(SECTOR_ETFS.values())

# Color palette for sectors
SECTOR_COLORS = {
    'XLC':  '#FF6B6B',   # Communication - Red
    'XLY':  '#FFA07A',   # Consumer Disc - Salmon
    'XLP':  '#98D8C8',   # Consumer Staples - Teal
    'XLF':  '#7B68EE',   # Financials - Purple
    'XLE':  '#FFD700',   # Energy - Gold
    'XLV':  '#87CEEB',   # Health Care - Sky Blue
    'XLI':  '#DDA0DD',   # Industrials - Plum
    'XLK':  '#00CED1',   # Tech - Teal
    'XLB':  '#F0E68C',   # Materials - Khaki
    'XLRE': '#BC8F8F',   # Real Estate - Rosy
    'XLU':  '#90EE90',   # Utilities - Light Green
}


def build_networkx_graph(adj_matrix: np.ndarray,
                         node_features: np.ndarray = None,
                         directed: bool = True,
                         edge_threshold: float = 0.05) -> nx.Graph:
    """
    Build a networkx graph from adjacency matrix and node features.
    
    Args:
        adj_matrix: Edge weight matrix (11, 11)
        node_features: Node feature vector (11, n_features), optional
        directed: Whether to create a directed graph (Module 2)
        edge_threshold: Minimum edge weight to include
        
    Returns:
        networkx Graph or DiGraph
    """
    G = nx.DiGraph() if directed else nx.Graph()

    # Add nodes
    for i, sector in enumerate(SECTORS):
        attrs = {
            'name': SECTOR_NAMES[i],
            'ticker': sector,
            'color': SECTOR_COLORS[sector],
        }
        if node_features is not None and i < len(node_features):
            attrs.update({
                'returns': float(node_features[i, 0]),
                'volatility': float(node_features[i, 1]),
                'momentum': float(node_features[i, 2]),
                'valuation': float(node_features[i, 3]),
                'comm_beta': float(node_features[i, 4]),
            })
        G.add_node(sector, **attrs)

    # Add edges
    for i in range(len(SECTORS)):
        for j in range(len(SECTORS)):
            if i != j and abs(adj_matrix[i, j]) > edge_threshold:
                G.add_edge(SECTORS[i], SECTORS[j],
                          weight=float(adj_matrix[i, j]),
                          abs_weight=float(abs(adj_matrix[i, j])))

    return G


def graph_to_cytoscape_elements(G: nx.Graph,
                                 node_size_feature: str = 'volatility',
                                 highlight_nodes: list = None) -> list:
    """
    Convert networkx graph to dash-cytoscape element format.
    
    Args:
        G: networkx graph
        node_size_feature: Which node feature determines node size
        highlight_nodes: List of node IDs to highlight
        
    Returns:
        List of cytoscape element dicts
    """
    elements = []

    # Nodes
    for node, data in G.nodes(data=True):
        size = data.get(node_size_feature, 0.5) * 40 + 20
        classes = 'highlighted' if highlight_nodes and node in highlight_nodes else ''

        elements.append({
            'data': {
                'id': node,
                'label': f"{node}\n{data.get('name', '')}",
                'name': data.get('name', node),
                'ticker': data.get('ticker', node),
                'size': size,
                'color': data.get('color', '#888'),
                'returns': data.get('returns', 0),
                'volatility': data.get('volatility', 0),
                'momentum': data.get('momentum', 0),
                'valuation': data.get('valuation', 0),
                'comm_beta': data.get('comm_beta', 0),
            },
            'classes': classes,
        })

    # Edges
    for u, v, data in G.edges(data=True):
        weight = data.get('abs_weight', 0.1)
        raw_weight = data.get('weight', 0)

        elements.append({
            'data': {
                'source': u,
                'target': v,
                'weight': weight,
                'raw_weight': raw_weight,
                'width': max(weight * 5, 0.5),
                'color': '#2ECC71' if raw_weight > 0 else '#E74C3C',
            }
        })

    return elements


def build_plotly_graph(G: nx.Graph, title: str = '') -> go.Figure:
    """
    Build a Plotly figure of the sector graph for standalone rendering.
    
    Args:
        G: networkx graph
        title: Plot title
        
    Returns:
        Plotly Figure
    """
    # Use spring layout for positioning
    pos = nx.spring_layout(G, seed=42, k=2.0)

    # Edge traces
    edge_x, edge_y = [], []
    edge_colors = []
    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='rgba(150,150,150,0.5)'),
        hoverinfo='none',
        mode='lines'
    )

    # Node traces
    node_x = [pos[n][0] for n in G.nodes()]
    node_y = [pos[n][1] for n in G.nodes()]
    node_text = []
    node_colors = []
    node_sizes = []

    for node, data in G.nodes(data=True):
        text = (f"{data.get('name', node)} ({node})<br>"
                f"Returns: {data.get('returns', 0):.4f}<br>"
                f"Volatility: {data.get('volatility', 0):.3f}<br>"
                f"Momentum: {data.get('momentum', 0):.3f}")
        node_text.append(text)
        node_colors.append(data.get('color', '#888'))
        node_sizes.append(data.get('volatility', 0.3) * 40 + 15)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=[n for n in G.nodes()],
        textposition='top center',
        hovertext=node_text,
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=2, color='white'),
        )
    )

    fig = go.Figure(data=[edge_trace, node_trace])

    fig.update_layout(
        title=dict(text=title or 'Sector Knowledge Graph', font=dict(size=16)),
        showlegend=False,
        hovermode='closest',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=500,
        margin=dict(l=20, r=20, t=60, b=20),
        plot_bgcolor='white',
    )

    return fig


def build_graph_diff(G1: nx.Graph, G2: nx.Graph,
                     title: str = 'Graph Diff') -> go.Figure:
    """
    Module 9: Graph Diff View.
    Show edge weight changes between two graph snapshots.
    
    Green = strengthened edges
    Red = weakened edges
    Gray = unchanged
    
    Args:
        G1: First graph snapshot (earlier)
        G2: Second graph snapshot (later)
        title: Plot title
        
    Returns:
        Plotly Figure
    """
    pos = nx.spring_layout(G1, seed=42, k=2.0)

    # Compute edge weight changes
    edge_traces = []
    for u, v, d2 in G2.edges(data=True):
        if G1.has_edge(u, v):
            d1 = G1[u][v]
            delta = d2.get('weight', 0) - d1.get('weight', 0)
        else:
            delta = d2.get('weight', 0)

        x0, y0 = pos[u]
        x1, y1 = pos[v]

        if delta > 0.02:
            color = 'green'
            width = delta * 10
        elif delta < -0.02:
            color = 'red'
            width = abs(delta) * 10
        else:
            color = 'gray'
            width = 0.5

        edge_traces.append(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            mode='lines',
            line=dict(width=max(width, 0.5), color=color),
            hoverinfo='text',
            hovertext=f'{u}→{v}: Δ={delta:+.3f}',
            showlegend=False
        ))

    # Node trace
    node_x = [pos[n][0] for n in G2.nodes()]
    node_y = [pos[n][1] for n in G2.nodes()]
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=[n for n in G2.nodes()],
        textposition='top center',
        marker=dict(
            size=20,
            color=[SECTOR_COLORS.get(n, '#888') for n in G2.nodes()],
            line=dict(width=2, color='white')
        ),
        showlegend=False
    )

    fig = go.Figure(data=edge_traces + [node_trace])

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=500,
        margin=dict(l=20, r=20, t=60, b=20),
        plot_bgcolor='white',
        annotations=[
            dict(x=0.01, y=0.99, xref='paper', yref='paper',
                 text='🟢 Strengthened  🔴 Weakened  ⚪ Unchanged',
                 showarrow=False, font=dict(size=12)),
        ]
    )

    return fig


if __name__ == '__main__':
    from data.bea_io import build_io_matrix

    adj = build_io_matrix()
    node_feats = np.random.rand(11, 5)

    G = build_networkx_graph(adj, node_feats)
    print(f'Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges')

    elements = graph_to_cytoscape_elements(G)
    print(f'Cytoscape elements: {len(elements)}')

    fig = build_plotly_graph(G)
    fig.show()
