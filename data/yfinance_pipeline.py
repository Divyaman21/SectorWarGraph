from __future__ import annotations
"""
yfinance pipeline for sector ETF price data.
Pulls daily OHLCV data for all 11 SPDR sector ETFs and WTI crude oil futures.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import SECTOR_ETFS, OIL_TICKER, START_DATE, END_DATE
from utils.logger import get_logger, log_step, log_dataframe_info

logger = get_logger('data.yfinance_pipeline')


def fetch_sector_prices(start: str = START_DATE,
                        end: str = None) -> pd.DataFrame:
    """
    Fetch daily closing prices for all 11 SPDR sector ETFs.
    
    Args:
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD), defaults to today
        
    Returns:
        DataFrame with columns as ticker symbols, indexed by date
    """
    log_step(logger, 'Fetching sector ETF prices',
             f'Tickers: {list(SECTOR_ETFS.keys())}')

    if end is None:
        end = datetime.now().strftime('%Y-%m-%d')

    tickers = list(SECTOR_ETFS.keys())

    try:
        df = yf.download(tickers, start=start, end=end,
                         auto_adjust=True, progress=False)
        # Handle MultiIndex columns from yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df = df['Close']
        elif 'Close' in df.columns:
            df = df[['Close']]
            df.columns = tickers

        df = df.dropna(how='all')
        log_dataframe_info(logger, df, 'Sector prices')
        return df

    except Exception as e:
        logger.warning(f'yfinance failed ({e}), generating synthetic prices')
        return _generate_synthetic_prices(start, end, tickers)


def fetch_oil_prices(start: str = START_DATE,
                     end: str = None) -> pd.Series:
    """
    Fetch WTI crude oil futures daily closing prices.
    
    Returns:
        Series of oil closing prices indexed by date
    """
    log_step(logger, 'Fetching oil prices', f'Ticker: {OIL_TICKER}')

    if end is None:
        end = datetime.now().strftime('%Y-%m-%d')

    try:
        oil = yf.download(OIL_TICKER, start=start, end=end,
                          auto_adjust=True, progress=False)
        if isinstance(oil.columns, pd.MultiIndex):
            series = oil['Close'].iloc[:, 0]
        elif 'Close' in oil.columns:
            series = oil['Close']
        else:
            series = oil.iloc[:, 0]
        series.name = 'WTI'
        logger.info(f'Oil prices: {len(series)} trading days')
        return series

    except Exception as e:
        logger.warning(f'Oil price fetch failed ({e}), generating synthetic')
        return _generate_synthetic_oil(start, end)


def fetch_oil_returns(start: str = START_DATE,
                      end: str = None) -> pd.Series:
    """Fetch oil price returns (daily pct change)."""
    prices = fetch_oil_prices(start, end)
    returns = prices.pct_change().dropna()
    returns.name = 'WTI_returns'
    return returns


def compute_sector_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily returns from price data."""
    returns = prices.pct_change().dropna()
    log_dataframe_info(logger, returns, 'Sector returns')
    return returns


def compute_monthly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute monthly returns from daily prices."""
    monthly = prices.resample('ME').last()
    returns = monthly.pct_change().dropna()
    log_dataframe_info(logger, returns, 'Monthly returns')
    return returns


def get_iv_skew(ticker: str, expiry_weeks: int = 4) -> float:
    """
    Compute put/call implied volatility skew for a sector ETF.
    This captures tail-risk pricing that realized vol misses.
    
    Args:
        ticker: ETF ticker symbol
        expiry_weeks: Target option expiry in weeks
        
    Returns:
        Skew ratio (OTM put IV / ATM call IV)
    """
    try:
        t = yf.Ticker(ticker)
        options_dates = t.options
        if not options_dates:
            return 1.0

        exp = options_dates[min(expiry_weeks - 1, len(options_dates) - 1)]
        chain = t.option_chain(exp)

        puts = chain.puts[['strike', 'impliedVolatility']]
        calls = chain.calls[['strike', 'impliedVolatility']]

        current_price = t.info.get('currentPrice') or t.info.get('regularMarketPrice', 100)

        # ATM call IV
        atm_idx = (calls['strike'] - current_price).abs().argsort()[:1]
        atm_call_iv = calls.iloc[atm_idx]['impliedVolatility'].values[0]

        # OTM put IV (strikes below 95% of current price)
        otm_puts = puts[puts['strike'] < current_price * 0.95]
        otm_put_iv = otm_puts['impliedVolatility'].mean()

        if atm_call_iv > 0 and not np.isnan(otm_put_iv):
            return otm_put_iv / atm_call_iv
        return 1.0

    except Exception as e:
        logger.warning(f'IV skew computation failed for {ticker}: {e}')
        return 1.0


def _generate_synthetic_prices(start: str, end: str,
                                tickers: list[str]) -> pd.DataFrame:
    """Generate synthetic price data for development."""
    logger.info('Generating synthetic price data')
    np.random.seed(42)

    dates = pd.bdate_range(start, end)  # business days
    n = len(dates)

    # Base prices and volatilities for each sector
    base_prices = {
        'XLC': 65, 'XLY': 170, 'XLP': 75, 'XLF': 38, 'XLE': 85,
        'XLV': 135, 'XLI': 105, 'XLK': 190, 'XLB': 80, 'XLRE': 38, 'XLU': 65
    }
    vols = {
        'XLC': 0.015, 'XLY': 0.013, 'XLP': 0.008, 'XLF': 0.012, 'XLE': 0.020,
        'XLV': 0.010, 'XLI': 0.011, 'XLK': 0.016, 'XLB': 0.014, 'XLRE': 0.013, 'XLU': 0.009
    }

    data = {}
    for ticker in tickers:
        base = base_prices.get(ticker, 100)
        vol = vols.get(ticker, 0.012)
        returns = np.random.normal(0.0003, vol, n)
        # Add some war shock events
        shock_days = np.random.choice(n, size=10, replace=False)
        returns[shock_days] += np.random.normal(-0.02, 0.01, 10)
        prices = base * np.cumprod(1 + returns)
        data[ticker] = prices

    df = pd.DataFrame(data, index=dates[:n])
    df.index.name = 'Date'
    return df


def _generate_synthetic_oil(start: str, end: str) -> pd.Series:
    """Generate synthetic oil price data."""
    np.random.seed(44)
    dates = pd.bdate_range(start, end)
    returns = np.random.normal(0.0005, 0.025, len(dates))
    prices = 80 * np.cumprod(1 + returns)
    series = pd.Series(prices, index=dates, name='WTI')
    return series


if __name__ == '__main__':
    prices = fetch_sector_prices()
    print(f'\nSector prices shape: {prices.shape}')
    print(prices.tail())

    oil = fetch_oil_prices()
    print(f'\nOil prices: {len(oil)} days')
    print(oil.tail())
