# 🌍 Sector War Graph
### Developed by **Divyaman Joshi**
### Middle East War Impact on Sector Knowledge Graph
**Temporal GNN-based analysis of how geopolitical events propagate through equity sectors**

---

## 🚀 Quick Start

```bash
cd SectorWarGraph-main

# Install dependencies
pip3 install -r requirements.txt

# Setup your credentials
# 1. Copy 'config_template.py' to a new file named 'config.py'
# 2. Open 'config.py' and enter your ACLED email and password.
# (Note: 'config.py' is ignored by Git to keep your passwords safe!)

# Launch interactive dashboard (demo mode, no API keys needed)
python3 run_dashboard.py

# Run the full pipeline with real data (requires ACLED API key)
python3 main.py --train --epochs 100
```

Then open **://127.0.0.1:8050** in your browser.

---

## 📁 Project Structure

```
sector_war_graph/
├── main.py                      # Entry point — orchestrates full pipeline
├── config.py                    # API keys, hyperparameters, constants
├── requirements.txt             # All pip dependencies
├── run_dashboard.py             # Quick dashboard launcher
│
├── data/
│   ├── acled_pipeline.py        # ACLED conflict event ingestion + classification
│   ├── gdelt_pipeline.py       http # GDELT news headline ingestion + tone scoring
│   ├── yfinance_pipeline.py     # Sector ETF prices + oil price data
│   └── bea_io.py                # BEA I-O table + supply-chain matrix
│
├── features/
│   ├── sensitivity_matrix.py    # ★ Key file: 15×11 event-sector sensitivity matrix
│   ├── node_features.py         # 5 node features per sector
│   ├── edge_features.py         # 4 edge features (corr, I-O, oil, supply)
│   └── event_encoder.py         # FinBERT/fallback event embedding pipeline
│
├── model/
│   ├── temporal_gnn.py          # T-GNN: GATv2Conv + GRU temporal dynamics
│   ├── regime_detector.py       # HMM-based war regime classifier
│   └── counterfactual.py        # What-if simulator + shock propagation
│
├── viz/
│   ├── dashboard.py             # Full Dash interactive dashboard
│   ├── graph_renderer.py        # Networkx → Cytoscape/Plotly rendering
│   └── heatmap.py               # Sector rotation heatmap + sensitivity viz
│
└── utils/
    ├── data_store.py            # Parquet-based caching
    └── logger.py                # Structured colored logging
```

---

## 🧠 Architecture

### Sensitivity Matrix (15 events × 11 sectors)
The core of the system. Maps geopolitical events to sector impacts using a 3-phase approach:
1. **Phase 1** — Hand-crafted domain knowledge priors
2. **Phase 2** — Empirical calibration from event-return correlations  
3. **Phase 3** — MultiTaskLasso learned sensitivity (sparse, data-driven)

### Temporal GNN Model
```
Input: (T months, 11 sectors, 5 node features)
      + (T months, 11×11, 4 edge features)
      + (T months, 768-dim event embeddings)

Layer 1: Linear → ReLU         (5 → 64)
Layer 2: EdgeMLP               (4 → 16)
Layer 3: GATv2Conv × 2         (64 → 128, attention-weighted)
Layer 4: GRU Cell              (temporal dynamics)
Layer 5: Event Fusion          (128 + 768 → 128)
Layer 6: Predictor             (128 → 1) per sector

Output: Next-month sector return predictions (T, 11)
```

### 5 Node Features Per Sector
| Feature | Formula | Normalization |
|---------|---------|---------------|
| Returns | Rolling 30d mean return | Z-score |
| Volatility | 30d std × √252 | Min-max |
| Momentum | Price[t]/Price[t-30] - 1 | Z-score |
| Valuation | Price / 252d MA | Min-max |
| Oil Beta | Corr(sector, WTI) | Raw -1 to 1 |

### 4 Edge Features
| Feature | Type | Shape |
|---------|------|-------|
| Rolling correlation | Dynamic | T×11×11 |
| I-O dependence | Static | 11×11 |
| Oil co-exposure | Dynamic | T×11×11 |
| Supply-chain | Semi-static | 11×11 |

---

## 📊 Dashboard Panels

| Panel | Description |
|-------|-------------|
| 🌐 Sector Graph | Force-directed cytoscape graph, color-coded nodes |
| 📅 Timeline Slider | Oct 2023 → present, monthly snapshots |
| 🔽 Edge Mode | Toggle: Correlation / I-O / Oil / Supply Chain |
| 🔍 Node Inspector | Feature bar chart for selected sector |
| 🗺️ Rotation Heatmap | 11×T z-score heatmap with event markers |
| 🔮 What-If Panel | Oil price slider → sector impact delta |
| 📰 Event Feed | Scrollable ACLED/GDELT events |
| 🏷️ Regime Badge | Current war regime (Escalation/Plateau/De-escalation) |

---

## 🔧 Configuration

Edit `config.py` to set:
```python
ACLED_KEY   = 'YOUR_KEY_HERE'    # acleddata.com (free registration)
ACLED_EMAIL = 'your@email.com'
START_DATE  = '2023-10-01'
END_DATE    = '2026-03-28'
```

**Without API keys**: The system auto-generates synthetic ACLED/GDELT data for development and demo purposes.

---

## 🗂️ 10 Improvement Modules

| # | Module | Location |
|---|--------|----------|
| 1 | HMM Regime Detector | `model/regime_detector.py` |
| 2 | Directed Asymmetric Graph | `viz/graph_renderer.py` |
| 3 | Counterfactual Simulator | `model/counterfactual.py` |
| 4 | Sentiment Co-movement Edge | `features/edge_features.py` |
| 5 | Options IV Skew Feature | `data/yfinance_pipeline.py` |
| 6 | Fund Flow Node Feature | (stub in `features/node_features.py`) |
| 7 | Shock Propagation Sim | `model/counterfactual.py` |
| 8 | Sector Rotation Heatmap | `viz/heatmap.py` |
| 9 | Graph Diff View | `viz/graph_renderer.py` |
| 10 | GAT Attention Explainability | `model/temporal_gnn.py` |

---

## 📦 Dependencies

```
yfinance, pandas, numpy, scikit-learn, hmmlearn
networkx, requests, dash, dash-cytoscape, plotly, pyarrow
torch (optional, for GNN training)
transformers (optional, for FinBERT embeddings)
torch_geometric (optional, for GATv2Conv)
```
