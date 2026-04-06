from __future__ import annotations
"""
GDELT (Global Database of Events, Language, and Tone) pipeline.
Fetches news event headlines related to MENA conflicts using gdeltdoc.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import MENA_KEYWORDS, START_DATE, END_DATE
from utils.logger import get_logger, log_step, log_dataframe_info

logger = get_logger('data.gdelt_pipeline')


def fetch_gdelt_headlines(keyword_list: list[str] = None,
                          start_date: str = START_DATE,
                          end_date: str = None,
                          num_records: int = 250) -> pd.DataFrame:
    """
    Fetch GDELT news headlines related to MENA conflict events.
    
    Args:
        keyword_list: Search keywords (defaults to MENA_KEYWORDS)
        start_date: Start date string
        end_date: End date string
        num_records: Max records per query
        
    Returns:
        DataFrame with article metadata
    """
    log_step(logger, 'Fetching GDELT headlines',
             f'Keywords: {len(keyword_list or MENA_KEYWORDS)} terms')

    if keyword_list is None:
        keyword_list = MENA_KEYWORDS
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    try:
        import time
        from gdeltdoc import GdeltDoc, Filters

        gd = GdeltDoc()
        all_articles = []

        for i, keyword in enumerate(keyword_list):
            if i > 0:
                logger.info('  Waiting 5 seconds to respect GDELT rate limit...')
                time.sleep(5)  # Mandatory GDELT API rate limit delay

            try:
                f = Filters(
                    keyword=keyword,
                    start_date=start_date,
                    end_date=end_date,
                    num_records=num_records
                )
                articles = gd.article_search(f)
                if articles is not None and not articles.empty:
                    articles['search_keyword'] = keyword
                    all_articles.append(articles)
                    logger.info(f'  "{keyword}": {len(articles)} articles')
                else:
                    logger.warning(f'  "{keyword}": no articles found')
            except Exception as e:
                logger.warning(f'  "{keyword}" failed: {e}')
                if "limit requests" in str(e):
                    time.sleep(2) # Extra buffer for rate limits

        if all_articles:
            df = pd.concat(all_articles, ignore_index=True)
            df = _process_gdelt(df)
            log_dataframe_info(logger, df, 'GDELT combined')
            return df
        else:
            raise RuntimeError(f'No GDELT headlines found for any of the {len(keyword_list)} keywords.')

    except Exception as e:
        logger.error(f'GDELT fetch failed: {e}')
        raise


def _process_gdelt(df: pd.DataFrame) -> pd.DataFrame:
    """Process raw GDELT article data."""
    # Standardize column names
    col_map = {
        'url': 'url',
        'title': 'title',
        'seendate': 'event_date',
        'domain': 'domain',
        'language': 'language',
        'socialimage': 'image_url',
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # Parse dates
    if 'event_date' in df.columns:
        df['event_date'] = pd.to_datetime(df['event_date'], errors='coerce')
    else:
        df['event_date'] = pd.Timestamp.now()

    # Drop duplicates by title
    df = df.drop_duplicates(subset=['title'], keep='first')

    # Compute a pseudo tone score from title sentiment keywords
    df['tone_score'] = df['title'].apply(_estimate_tone) if 'title' in df.columns else 0.0

    return df.sort_values('event_date').reset_index(drop=True)


def _estimate_tone(title: str) -> float:
    """
    Estimate a GDELT-style tone score from headline text.
    Negative = negative tone, Positive = positive tone.
    Range approximately -10 to +10.
    """
    if not isinstance(title, str):
        return 0.0

    title_lower = title.lower()

    negative_words = [
        'war', 'attack', 'strike', 'bomb', 'kill', 'dead', 'death',
        'missile', 'explosion', 'threat', 'conflict', 'crisis',
        'violence', 'terror', 'destroy', 'casualties', 'invasion',
        'escalat', 'retaliat', 'sanction', 'blockade', 'siege'
    ]
    positive_words = [
        'peace', 'ceasefire', 'deal', 'agreement', 'negotiat',
        'diplomatic', 'aid', 'humanitarian', 'relief', 'truce',
        'dialogue', 'cooperat', 'de-escalat', 'withdraw'
    ]

    neg_count = sum(1 for w in negative_words if w in title_lower)
    pos_count = sum(1 for w in positive_words if w in title_lower)

    score = (pos_count - neg_count) * 2.5
    return max(-10.0, min(10.0, score))


def _generate_synthetic_gdelt(start: str, end: str) -> pd.DataFrame:
    """Generate synthetic GDELT-like headline data for development."""
    logger.info('Generating synthetic GDELT data for development')
    np.random.seed(43)

    dates = pd.date_range(start, end, freq='D')

    headlines = [
        'Israel launches airstrikes on Gaza amid escalating conflict',
        'Houthi rebels fire missiles at Red Sea shipping routes',
        'Iran warns of retaliation following military strikes',
        'Oil prices surge as Middle East tensions mount',
        'Ceasefire talks between Israel and Hamas show progress',
        'Hezbollah launches rockets into northern Israel',
        'US deploys carrier group to Eastern Mediterranean',
        'Suez Canal traffic disrupted by Houthi attacks',
        'Iran nuclear program advances despite sanctions',
        'OPEC announces production cuts amid geopolitical risks',
        'Humanitarian crisis deepens in Gaza strip',
        'Diplomatic efforts intensify for Middle East peace',
        'Cyber attacks target critical infrastructure in region',
        'Israel begins ground operation in southern Lebanon',
        'Saudi Arabia mediates regional de-escalation talks',
        'Strait of Hormuz shipping insurance rates spike',
        'UN warns of famine in conflict zones',
        'Defense stocks rally on increased military spending',
        'Oil tanker attacked in Gulf of Oman',
        'Peace agreement signed between warring factions',
    ]

    n = len(dates) * 2  # ~2 headlines per day
    df = pd.DataFrame({
        'event_date': np.random.choice(dates, n),
        'title': np.random.choice(headlines, n),
        'url': [f'https://news.example.com/article/{i}' for i in range(n)],
        'domain': np.random.choice(['reuters.com', 'aljazeera.com',
                                     'bbc.co.uk', 'cnn.com'], n),
        'language': 'English',
        'search_keyword': np.random.choice(MENA_KEYWORDS, n),
    })
    df['event_date'] = pd.to_datetime(df['event_date'])
    df['tone_score'] = df['title'].apply(_estimate_tone)
    df = df.sort_values('event_date').reset_index(drop=True)

    logger.info(f'Generated {len(df)} synthetic GDELT headlines')
    return df


if __name__ == '__main__':
    df = fetch_gdelt_headlines()
    print(f'\nFetched {len(df)} GDELT headlines')
    print(df[['event_date', 'title', 'tone_score']].head(20))
