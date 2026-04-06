from __future__ import annotations
"""
Temporal Graph Neural Network (T-GNN) with attention-based edge weighting.

Architecture:
  1. Linear + ReLU     — Project node features (11,5) → (11,64)
  2. Edge MLP          — Encode edge features (N_edges,4) → (N_edges,16)
  3. GATv2Conv x2      — Attention-weighted graph agg → (11,128)
  4. GRU Cell           — Temporal dynamics across snapshots
  5. Event Fusion       — Inject war event signal
  6. Linear             — Predict next-month sector return (11,1)
"""

import torch
import torch.nn as nn
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import GNN_HIDDEN, GNN_EPOCHS, GNN_LR, GNN_WEIGHT_DECAY, GNN_DROPOUT, GNN_HEADS, EVENT_EMB_DIM
from utils.logger import get_logger, log_step

logger = get_logger('model.temporal_gnn')

# Check if PyG is available
try:
    from torch_geometric.nn import GATv2Conv
    from torch_geometric.data import Data
    HAS_PYG = True
except ImportError:
    logger.warning('torch_geometric not installed, using fallback GNN implementation')
    HAS_PYG = False


class EdgeMLP(nn.Module):
    """MLP to encode edge features into edge embeddings."""

    def __init__(self, in_dim: int = 4, out_dim: int = 16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(),
            nn.Linear(32, out_dim),
            nn.ReLU()
        )

    def forward(self, edge_attr: torch.Tensor) -> torch.Tensor:
        return self.mlp(edge_attr)


class SectorWarGNN(nn.Module):
    """
    Temporal Graph Attention Network for sector war impact modeling.
    
    Processes snapshot sequences using GRU for temporal dynamics,
    with attention-weighted edge aggregation via GATv2Conv.
    """

    def __init__(self, node_feat_dim: int = 5, edge_feat_dim: int = 4,
                 hidden: int = GNN_HIDDEN, event_emb_dim: int = EVENT_EMB_DIM,
                 n_heads: int = GNN_HEADS, dropout: float = GNN_DROPOUT):
        super().__init__()

        self.hidden = hidden
        self.n_sectors = 11

        # ── Node feature projection ──────────────────────────────────────
        self.node_proj = nn.Sequential(
            nn.Linear(node_feat_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64)
        )

        # ── Edge feature encoding ────────────────────────────────────────
        self.edge_proj = EdgeMLP(edge_feat_dim, 16)

        # ── Graph Attention layers ───────────────────────────────────────
        if HAS_PYG:
            self.gat1 = GATv2Conv(64, 64, heads=n_heads, edge_dim=16,
                                  concat=True, add_self_loops=True)
            self.gat2 = GATv2Conv(64 * n_heads, hidden, heads=1, edge_dim=16,
                                  concat=False, add_self_loops=True)
        else:
            # Fallback: simple message-passing with attention
            self.gat1 = FallbackGAT(64, 64 * n_heads, 16)
            self.gat2 = FallbackGAT(64 * n_heads, hidden, 16)

        # ── Temporal GRU ─────────────────────────────────────────────────
        self.gru = nn.GRUCell(hidden, hidden)

        # ── Event fusion ─────────────────────────────────────────────────
        self.event_fusion = nn.Sequential(
            nn.Linear(hidden + event_emb_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # ── Prediction head ──────────────────────────────────────────────
        self.predictor = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1)
        )

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden)

        # ── Attention weight storage (Module 10) ─────────────────────────
        self.attention_weights = []

    def forward(self, snapshots: list, event_embs: torch.Tensor,
                hx: torch.Tensor = None,
                store_attention: bool = False):
        """
        Forward pass through temporal graph snapshots.
        
        Args:
            snapshots: List of PyG Data objects (or dicts for fallback)
            event_embs: Event embeddings tensor (T, event_emb_dim)
            hx: Initial hidden state (N, hidden)
            store_attention: Whether to store GAT attention weights
            
        Returns:
            Predictions tensor (T, N, 1)
        """
        N = self.n_sectors
        if hx is None:
            hx = torch.zeros(N, self.hidden, device=event_embs.device)

        self.attention_weights = []
        preds = []

        for t, graph in enumerate(snapshots):
            if HAS_PYG:
                x = self.node_proj(graph.x)
                ea = self.edge_proj(graph.edge_attr)

                if store_attention:
                    x1, (edge_idx, alpha1) = self.gat1(
                        x, graph.edge_index, ea,
                        return_attention_weights=True)
                    x1 = self.dropout(torch.relu(x1))
                    x2, (edge_idx2, alpha2) = self.gat2(
                        x1, graph.edge_index, ea,
                        return_attention_weights=True)
                    x2 = self.dropout(torch.relu(x2))
                    self.attention_weights.append({
                        'gat1_alpha': alpha1.detach().cpu(),
                        'gat2_alpha': alpha2.detach().cpu(),
                        'edge_index': edge_idx.detach().cpu()
                    })
                else:
                    x1 = self.dropout(torch.relu(
                        self.gat1(x, graph.edge_index, ea)))
                    x2 = self.dropout(torch.relu(
                        self.gat2(x1, graph.edge_index, ea)))
            else:
                x = self.node_proj(graph['x'])
                ea = self.edge_proj(graph['edge_attr'])
                x1 = self.dropout(torch.relu(
                    self.gat1(x, graph['edge_index'], ea)))
                x2 = self.dropout(torch.relu(
                    self.gat2(x1, graph['edge_index'], ea)))

            # Temporal update
            x2 = self.layer_norm(x2)
            hx = self.gru(x2, hx)

            # Event fusion
            ev = event_embs[t].unsqueeze(0).expand(N, -1)
            fused = self.event_fusion(torch.cat([hx, ev], dim=-1))

            # Prediction
            pred = self.predictor(fused)  # (N, 1)
            preds.append(pred)

        return torch.stack(preds, dim=0)  # (T, N, 1)

    def get_attention_weights(self):
        """Return stored attention weights for explainability (Module 10)."""
        return self.attention_weights


class FallbackGAT(nn.Module):
    """
    Simple attention-based message passing for when PyG is not installed.
    Approximates GATv2Conv behavior.
    """

    def __init__(self, in_dim: int, out_dim: int, edge_dim: int):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim)
        self.attn = nn.Linear(2 * out_dim + edge_dim, 1)
        self.out_dim = out_dim

    def forward(self, x, edge_index, edge_attr=None,
                return_attention_weights=False):
        h = self.W(x)
        src, dst = edge_index[0], edge_index[1]

        # Compute attention scores
        src_h = h[src]
        dst_h = h[dst]
        if edge_attr is not None:
            attn_input = torch.cat([src_h, dst_h, edge_attr], dim=-1)
        else:
            attn_input = torch.cat([src_h, dst_h,
                                     torch.zeros(src_h.size(0), 16)], dim=-1)

        alpha = torch.softmax(self.attn(attn_input).squeeze(-1), dim=0)

        # Aggregate messages
        out = torch.zeros_like(h)
        for i in range(edge_index.size(1)):
            out[dst[i]] += alpha[i] * src_h[i]

        if return_attention_weights:
            return out, (edge_index, alpha.unsqueeze(-1))
        return out


def build_pyg_snapshots(node_features: np.ndarray,
                        edge_features: np.ndarray,
                        edge_index: np.ndarray) -> list:
    """
    Convert numpy feature tensors into PyG Data objects (or dicts for fallback).
    
    Args:
        node_features: Shape (T, N, node_feat_dim)
        edge_features: Shape (T, N, N, edge_feat_dim)
        edge_index: Shape (2, n_edges)
        
    Returns:
        List of Data objects (or dicts)
    """
    T = node_features.shape[0]
    snapshots = []

    ei_tensor = torch.tensor(edge_index, dtype=torch.long)

    for t in range(T):
        x = torch.tensor(node_features[t], dtype=torch.float32)

        # Extract edge attributes per edge
        src, dst = edge_index[0], edge_index[1]
        ea = torch.tensor(edge_features[t, src, dst], dtype=torch.float32)

        if HAS_PYG:
            data = Data(x=x, edge_index=ei_tensor, edge_attr=ea)
            snapshots.append(data)
        else:
            snapshots.append({
                'x': x,
                'edge_index': ei_tensor,
                'edge_attr': ea
            })

    return snapshots


def train_model(model: SectorWarGNN,
                snapshots: list,
                event_embs: torch.Tensor,
                targets: torch.Tensor,
                epochs: int = GNN_EPOCHS,
                lr: float = GNN_LR) -> list[float]:
    """
    Train the T-GNN model.
    
    Args:
        model: SectorWarGNN instance
        snapshots: List of graph snapshots
        event_embs: Event embedding tensor (T, event_emb_dim)
        targets: Target returns tensor (T, N)
        epochs: Number of training epochs
        lr: Learning rate
        
    Returns:
        List of training losses per epoch
    """
    log_step(logger, 'Training T-GNN model',
             f'Epochs={epochs}, lr={lr}')

    opt = torch.optim.Adam(model.parameters(), lr=lr,
                           weight_decay=GNN_WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    loss_fn = nn.HuberLoss()  # Robust to outliers (war spikes)

    losses = []
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()

        preds = model(snapshots, event_embs)
        loss = loss_fn(preds.squeeze(-1), targets)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        scheduler.step()

        losses.append(loss.item())

        if epoch % 20 == 0:
            logger.info(f'Epoch {epoch:4d}: loss={loss.item():.6f}, '
                       f'lr={scheduler.get_last_lr()[0]:.2e}')

    logger.info(f'Training complete. Final loss: {losses[-1]:.6f}')
    return losses


def predict(model: SectorWarGNN,
            snapshots: list,
            event_embs: torch.Tensor,
            store_attention: bool = True) -> np.ndarray:
    """
    Run inference on the trained model.
    
    Returns:
        numpy array of predictions (T, N)
    """
    model.eval()
    with torch.no_grad():
        preds = model(snapshots, event_embs, store_attention=store_attention)
    return preds.squeeze(-1).numpy()


if __name__ == '__main__':
    # Quick test with random data
    T, N, F_node, F_edge = 12, 11, 5, 4
    n_edges = N * (N - 1)

    node_feats = np.random.randn(T, N, F_node).astype(np.float32)
    edge_feats = np.random.randn(T, N, N, F_edge).astype(np.float32)

    from features.edge_features import build_edge_index
    edge_idx = build_edge_index(N)

    snapshots = build_pyg_snapshots(node_feats, edge_feats, edge_idx)
    event_embs = torch.randn(T, EVENT_EMB_DIM)
    targets = torch.randn(T, N)

    model = SectorWarGNN()
    print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')

    losses = train_model(model, snapshots, event_embs, targets, epochs=50)
    preds = predict(model, snapshots, event_embs)
    print(f'Predictions shape: {preds.shape}')
