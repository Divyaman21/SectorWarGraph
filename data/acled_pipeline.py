from __future__ import annotations
"""
ACLED (Armed Conflict Location and Event Data) pipeline.
Fetches structured conflict event records for MENA countries.
Register at acleddata.com for a free API key.
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import ACLED_PASSWORD, ACLED_EMAIL, MENA_COUNTRIES, START_DATE, END_DATE
from utils.logger import get_logger, log_step, log_dataframe_info

logger = get_logger('data.acled_pipeline')

ACLED_OAUTH_URL = 'https://acleddata.com/oauth/token'
ACLED_BASE_URL = 'https://acleddata.com/api/acled/read'

# ── Event type mapping to our 15 war event categories ────────────────────────
ACLED_EVENT_MAP = {
    'Battles': 'military_strike_MENA',
    'Explosions/Remote violence': 'military_strike_MENA',
    'Violence against civilians': 'humanitarian_crisis',
    'Protests': 'humanitarian_crisis',
    'Riots': 'humanitarian_crisis',
    'Strategic developments': 'diplomatic_progress',
}

ACLED_SUBTYPE_MAP = {
    'Air/drone strike': 'military_strike_MENA',
    'Shelling/artillery/missile attack': 'houthi_missile',
    'Armed clash': 'military_strike_MENA',
    'Attack': 'israel_ground_op',
    'Suicide bomb': 'military_strike_MENA',
    'Chemical weapon': 'military_strike_MENA',
    'Abduction/forced disappearance': 'humanitarian_crisis',
    'Peaceful protest': 'humanitarian_crisis',
    'Excessive force against protesters': 'humanitarian_crisis',
    'Agreement': 'ceasefire_signal',
    'Arrests': 'sanctions_imposed',
    'Change to group/activity': 'hezbollah_escalation',
    'Headquarters or base established': 'israel_ground_op',
}


def fetch_acled(start: str = START_DATE, end: str = None,
                use_cache: bool = True) -> pd.DataFrame:
    """
    Fetch ACLED conflict events for MENA countries via OAuth.

    Raises RuntimeError immediately if authentication or the data
    fetch fails — no synthetic-data fallback.

    Args:
        start: Start date (YYYY-MM-DD)
        end:   End date   (YYYY-MM-DD), defaults to today
        use_cache: (reserved for future cache layer)

    Returns:
        DataFrame with classified conflict events
    """
    log_step(logger, 'Fetching ACLED data',
             f'Period: {start} to {end or "present"}')

    if end is None:
        end = datetime.now().strftime('%Y-%m-%d')

    # ── Step 1: OAuth token request ────────────────────────────────────────
    logger.info(f'Requesting ACLED OAuth token for user: {ACLED_EMAIL}')
    try:
        token_res = requests.post(
            ACLED_OAUTH_URL,
            data={
                'username': ACLED_EMAIL,
                'password': ACLED_PASSWORD,
                'client_id': 'acled',
                'grant_type': 'password',
            },
            timeout=30,
        )
        token_res.raise_for_status()
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(
            f'ACLED OAuth request failed — check network / credentials.\n'
            f'URL: {ACLED_OAUTH_URL}\nError: {exc}'
        ) from exc

    token_data = token_res.json()
    access_token = token_data.get('access_token')
    if not access_token:
        raise RuntimeError(
            f'ACLED OAuth response did not contain an access_token.\n'
            f'Response: {token_data}'
        )
    logger.info('ACLED OAuth token obtained successfully.')

    # ── Step 2: Fetch conflict data ────────────────────────────────────────
    headers = {'Authorization': f'Bearer {access_token}'}
    params = {
        'country': '|'.join(MENA_COUNTRIES),
        'event_date': f'{start}|{end}',
        'event_date_where': 'BETWEEN',
        'limit': 50000,
    }

    logger.info(f'Fetching ACLED events from {ACLED_BASE_URL}')
    try:
        res = requests.get(ACLED_BASE_URL, params=params, headers=headers, timeout=120)
        res.raise_for_status()
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(
            f'ACLED data fetch failed.\nURL: {ACLED_BASE_URL}\nError: {exc}'
        ) from exc

    payload = res.json()
    records = payload.get('data', [])
    if not records:
        raise RuntimeError(
            'ACLED returned an empty dataset for the requested date range '
            f'({start} \u2013 {end}) and countries. '
            f'API message: {payload.get("status", "(no status)")}'
        )
    df = pd.DataFrame(records)
    df['event_date'] = pd.to_datetime(df['event_date'])
    log_dataframe_info(logger, df, 'ACLED raw')

    # ── Step 3: Enrich ────────────────────────────────────────────────────
    df = _classify_events(df)
    df = _compute_severity(df)

    log_dataframe_info(logger, df, 'ACLED processed')
    return df


def _classify_events(df: pd.DataFrame) -> pd.DataFrame:
    """Map ACLED event types to our 15 war event categories."""
    df['war_event_type'] = df.apply(_map_event, axis=1)
    return df


def _map_event(row) -> str:
    """Map a single ACLED event row to a war event type."""
    # Try sub-event type first (more specific)
    sub = row.get('sub_event_type', '')
    if sub in ACLED_SUBTYPE_MAP:
        return ACLED_SUBTYPE_MAP[sub]

    # Country-specific overrides
    country = row.get('country', '')
    event = row.get('event_type', '')

    if country in ('Yemen',) and event in ('Explosions/Remote violence',):
        return 'houthi_missile'
    if country in ('Lebanon',) and event in ('Battles', 'Explosions/Remote violence'):
        return 'hezbollah_escalation'
    if country in ('Iran',):
        return 'us_iran_tension'

    # Fall back to general map
    return ACLED_EVENT_MAP.get(event, 'military_strike_MENA')


def _compute_severity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a severity score for each event based on fatalities and event type.
    Score range: 0.1 to 10.0
    """
    df['fatalities'] = pd.to_numeric(df.get('fatalities', 0), errors='coerce').fillna(0)

    # Log-scaled fatality component
    df['severity_score'] = np.log1p(df['fatalities'] + 1)

    # Event type multiplier
    type_weights = {
        'military_strike_MENA': 1.5,
        'houthi_missile': 1.8,
        'hezbollah_escalation': 2.0,
        'israel_ground_op': 2.0,
        'us_iran_tension': 2.5,
        'iran_nuclear_progress': 3.0,
        'oil_route_threat': 2.0,
        'oil_price_spike': 1.5,
        'shipping_disruption': 1.5,
        'sanctions_imposed': 1.0,
        'cyber_attack_MENA': 1.2,
        'humanitarian_crisis': 0.8,
        'ceasefire_signal': 0.5,
        'diplomatic_progress': 0.3,
        'opec_cut_announcement': 1.0,
    }
    df['severity_score'] *= df['war_event_type'].map(type_weights).fillna(1.0)
    df['severity_score'] = df['severity_score'].clip(0.1, 10.0)

    # Fatality bin for Lasso features
    df['fatality_bin'] = pd.cut(
        df['fatalities'], bins=[-1, 0, 5, 25, 100, 10000],
        labels=['none', 'low', 'medium', 'high', 'mass']
    )

    return df



if __name__ == '__main__':
    df = fetch_acled()
    print(f'\nFetched {len(df)} ACLED events')
    print(df[['event_date', 'country', 'war_event_type', 'severity_score']].head(20))
