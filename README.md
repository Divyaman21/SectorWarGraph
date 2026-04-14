# 🌍 Sector War Knowledge Graph
### Developed by **Divyaman Joshi**
**Mapping how geopolitical events propagate through equity sectors via a structural Knowledge Graph**

---

## 🚀 Quick Start

```bash
cd SectorWarGraph-main

# Install dependencies
pip3 install -r requirements.txt

# Setup your credentials in 'config.py' (ACLED email/password)

# Run the full pipeline and launch the dashboard
python3 main.py
```

Then open **http://127.0.0.1:8050** in your browser.

---

## 📁 Project Structure

```
sector_war_graph/
├── main.py                      # Entry point — orchestrates graph pipeline
├── config.py                    # API keys and constants
├── requirements.txt             # Pip dependencies (lightweight, no-GNN)
│
├── data/
│   ├── acled_pipeline.py        # ACLED conflict event ingestion
│   ├── gdelt_pipeline.py        # GDELT news headline ingestion
│   ├── yfinance_pipeline.py     # Sector ETF prices + oil price data
│   └── bea_io.py                # BEA I-O table + supply-chain matrix
│
├── features/
│   ├── sensitivity_matrix.py    # ★ Key file: 15×11 event-sector sensitivity matrix
│   ├── node_features.py         # 5 node features per sector
│   └── edge_features.py         # 4 edge features (corr, I-O, oil, supply)
│
├── model/
│   ├── regime_detector.py       # HMM-based war regime classifier
│   └── counterfactual.py        # What-if simulator + shock propagation
│
├── viz/
│   ├── dashboard.py             # Full Dash interactive dashboard
│   ├── graph_renderer.py        # Networkx → Cytoscape/Plotly rendering
│   └── heatmap.py               # Sector rotation heatmap + sensitivity viz
```

---

## 🧠 Architecture: The Knowledge Graph

### Structural Relationships
The core of the system is a **network representation** of the stock market. Instead of guessing prices, it maps the physical and economic reality of sectors:
1. **Economic Dependency**: Based on BEA Input-Output tables.
2. **Oil Sensitivity**: Real-time correlation between Middle East energy shocks and sector returns.
3. **Supply Chain Links**: Structural connections between materials, industrials, and technology.

### Graph Analytics
The system computes real-time network metrics:
*   **Sector Centrality**: Identifies which sector is the current "hub" or "bottleneck" in the network.
*   **Network Density**: Shows how tightly coupled the market becomes during crisis escalation.
*   **Regime HMM**: Automatically classifies the current period into *Escalation*, *Plateau*, or *De-escalation* based on event volatility.

---

## 📊 Dashboard Panels

| Panel | Description |
|-------|-------------|
| 🌐 Sector Graph | Force-directed knowledge graph of sector connections |
| 📈 Graph Analytics | Real-time Centrality, Density, and Average Degree metrics |
| 📅 Timeline Slider | Oct 2023 → present, monthly graph snapshots |
| 🔍 Node Inspector | Deep-dive into a sector's volatility, returns, and momentum |
| 🗺️ Rotation Heatmap | Market-wide visual of which sectors are rotating under stress |
| 🔮 What-If Panel | Simulate oil price shocks and view predicted graph-wide impact |
| 🏷️ Regime Badge | Current war regime indicator |

---

## 🔧 Configuration

Edit `config.py` to set:
```python
ACLED_EMAIL = 'your@email.com'
ACLED_PASSWORD = 'your_password'
START_DATE  = '2023-10-01'
END_DATE    = '2026-03-28'
```

---

## 📦 Core Dependencies

```
yfinance, pandas, numpy, scikit-learn, hmmlearn
networkx, requests, dash, dash-cytoscape, plotly, pyarrow
```
```
yfinance, pandas, numpy, scikit-learn, hmmlearn
networkx, requests, dash, dash-cytoscape, plotly, pyarrow
```
