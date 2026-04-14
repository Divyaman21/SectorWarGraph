from __future__ import annotations
"""
Configuration file for the Sector War Graph project.
All constants, API keys, and hyperparameters are defined here.
"""

# ── API Keys ──────────────────────────────────────────────────────────────────
ACLED_EMAIL = 'eez212381@iitd.ac.in'
ACLED_PASSWORD = 'Angira999@subhanu'

# ── Date Range ────────────────────────────────────────────────────────────────
START_DATE = '2023-10-01'
END_DATE = '2026-04-14'

# ── Feature Engineering ───────────────────────────────────────────────────────
WINDOW_DAYS = 30
FORWARD_DAYS = 21  # ~1 month for prediction target
LASSO_ALPHA = 0.01

# ── Regime Detection ─────────────────────────────────────────────────────────
N_REGIMES = 3
REGIME_LABELS = {0: 'Escalation', 1: 'Plateau', 2: 'De-escalation'}

# ── MENA Countries ───────────────────────────────────────────────────────────
MENA_COUNTRIES = [
    'Israel', 'Palestine', 'Lebanon', 'Yemen',
    'Syria', 'Iran', 'Iraq', 'Jordan', 'Egypt', 'Saudi Arabia'
]

# ── MENA Keywords for GDELT ──────────────────────────────────────────────────
MENA_KEYWORDS = [
    'Gaza war', 'Hamas attack', 'Israel Lebanon', 'Houthi Red Sea',
    'Iran nuclear', 'Hormuz strait', 'Rafah', 'Hezbollah', 'IRGC',
    'Israel airstrike', 'Yemen Houthi missile', 'Suez Canal disruption',
    'Syria Assad', 'Middle East oil', 'Iran sanctions'
]

# ── Sector ETF Mapping ───────────────────────────────────────────────────────
SECTOR_ETFS = {
    'XLC':  'Communication Services',
    'XLY':  'Consumer Discretionary',
    'XLP':  'Consumer Staples',
    'XLF':  'Financials',
    'XLE':  'Energy',
    'XLV':  'Health Care',
    'XLI':  'Industrials',
    'XLK':  'Information Technology',
    'XLB':  'Materials',
    'XLRE': 'Real Estate',
    'XLU':  'Utilities'
}

# ── BEA to GICS Concordance ─────────────────────────────────────────────────
BEA_TO_GICS = {
    'Oil and gas extraction; Petroleum refining': 'XLE',
    'Chemical manufacturing; Mining': 'XLB',
    'Machinery; Transportation equipment; Fabricated metals': 'XLI',
    'Computer & electronic products; Software publishers': 'XLK',
    'Broadcasting; Telecom; Publishing': 'XLC',
    'Food, beverage; Retail trade (staples)': 'XLP',
    'Food, beverage; Retail trade (discretionary)': 'XLY',
    'Ambulatory health; Hospitals; Pharmaceuticals': 'XLV',
    'Credit intermediation; Insurance; Securities': 'XLF',
    'Real estate; Rental & leasing': 'XLRE',
    'Electric power generation; Natural gas distribution': 'XLU',
}

# ── Oil Benchmark ────────────────────────────────────────────────────────────
OIL_TICKER = 'CL=F'  # WTI Crude Oil Futures

# ── Data Cache Directory ─────────────────────────────────────────────────────
CACHE_DIR = 'data/cache'

# ── Dashboard ────────────────────────────────────────────────────────────────
DASH_HOST = '127.0.0.1'
DASH_PORT = 8050
DASH_DEBUG = True
