"""
Market data download + local caching via yfinance.

Handles both MultiIndex layouts that different yfinance versions produce:
    (Ticker, Price)  or  (Price, Ticker)
and normalizes to (Ticker, Price).
"""

import os
import pickle
from typing import List

import pandas as pd
import yfinance as yf

from src.utils import config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _normalize_columns(raw: pd.DataFrame, assets: List[str]) -> pd.DataFrame:
    """
    Ensure columns are MultiIndex (Ticker, Price) with Price in
    {Open, High, Low, Close, Volume}.
    """
    if isinstance(raw.columns, pd.MultiIndex):
        lvl0 = list(raw.columns.get_level_values(0))
        lvl1 = list(raw.columns.get_level_values(1))
        # Newer yfinance: ('Price', 'Ticker'). Swap so ticker is outer.
        if 'Close' in lvl0 and 'Close' not in lvl1:
            raw = raw.swaplevel(0, 1, axis=1)
        return raw
    # Single ticker: build a MultiIndex manually.
    ticker = assets[0]
    raw.columns = pd.MultiIndex.from_product(
        [[ticker], raw.columns], names=['Ticker', 'Price']
    )
    return raw


def fetch_market_data(
    assets: List[str] = None,
    period: str = None,
    force_refresh: bool = False,
    cache_path: str = None,
) -> pd.DataFrame:
    assets = assets or config.DEFAULTS['assets']
    period = period or config.DEFAULTS['data_period']
    cache_path = cache_path or config.DEFAULTS['cache_path']

    # ---- Try cache first ----
    if not force_refresh and os.path.exists(cache_path):
        logger.info(f"Loading cached market data from {cache_path}")
        with open(cache_path, 'rb') as f:
            cached = pickle.load(f)
        if (isinstance(cached.columns, pd.MultiIndex)
                and 'Close' in cached.columns.get_level_values(1)
                and all(a in cached.columns.get_level_values(0) for a in assets)):
            return cached
        logger.warning("Cache invalid; re-downloading")

    # ---- Download ----
    logger.info(f"Downloading {assets} for period={period}")
    try:
        raw = yf.download(
            assets, period=period, auto_adjust=True,
            progress=False, threads=False,
        )

        if raw is None or raw.empty:
            raise RuntimeError("yfinance returned an empty DataFrame")

        data = _normalize_columns(raw, assets)

        fields = ['Open', 'High', 'Low', 'Close', 'Volume']
        keep = [(a, f) for a in assets for f in fields
                if (a, f) in data.columns]
        if not keep:
            raise RuntimeError(
                f"No usable (ticker, field) columns found. "
                f"Available: {list(data.columns)[:10]}"
            )
        data = data[keep].dropna(how='all')

        if data.empty or len(data) < 20:
            raise RuntimeError(
                f"Downloaded data too short ({len(data)} rows)."
            )

        os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)

        logger.info(f"Downloaded {len(data)} rows; cached to {cache_path}")
        return data

    except Exception as e:
        logger.error(f"Download failed: {e}")
        if os.path.exists(cache_path):
            logger.warning("Falling back to cache")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        raise


def get_close_prices(data: pd.DataFrame) -> pd.DataFrame:
    """Flat Close-price DataFrame: index=dates, columns=assets."""
    return data.xs('Close', axis=1, level=1, drop_level=True)


def get_volume(data: pd.DataFrame) -> pd.DataFrame:
    """Flat Volume DataFrame: index=dates, columns=assets."""
    return data.xs('Volume', axis=1, level=1, drop_level=True)