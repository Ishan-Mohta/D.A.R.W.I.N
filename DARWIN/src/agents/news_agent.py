"""
News sentiment agent backed by FinBERT (ProsusAI/finbert).

PIPELINE:
  1. Fetch recent RSS headlines for each asset (Yahoo Finance)
  2. Score each headline with FinBERT
  3. Aggregate into a per-asset sentiment score in [-1, +1]
  4. Emit BUY/SELL/HOLD based on thresholds (scaled by agent risk)

BACKTEST MODE:
  Historical news isn't fetched here. In backtest the agent looks up
  a precomputed sentiment_cache: {date_str: {ticker: score}}.
  If no cache, it defaults to HOLD — which is expected during historical
  runs. Live (today's date) fetches fresh headlines via FinBERT.

INTERFACE:
  adapt_parameters(self, performance_metrics, mutation_info=None)
  — matches the teammate's mutation system signature.
"""

import os
import pickle
from datetime import datetime
from typing import Dict, List, Tuple

import feedparser
import pandas as pd
import requests
from bs4 import BeautifulSoup

from src.agents.base_agent import BaseAgent, Signal
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
FINBERT_MODEL = 'ProsusAI/finbert'
YAHOO_RSS = ('https://feeds.finance.yahoo.com/rss/2.0/headline'
             '?s={ticker}&region=US&lang=en-US')

# In-memory cache so we don't re-score the same asset on the same day.
_SENTIMENT_CACHE: Dict[str, Tuple[float, List[Dict]]] = {}

# Lazy-loaded FinBERT pipeline (loaded on first use).
_FINBERT_PIPELINE = None


# ---------------------------------------------------------------------------
# FINBERT LOADER
# ---------------------------------------------------------------------------
def _get_finbert():
    """Lazy-load FinBERT pipeline. Caches it in module state."""
    global _FINBERT_PIPELINE
    if _FINBERT_PIPELINE is None:
        try:
            from transformers import pipeline
            logger.info("Loading FinBERT (first call — takes ~5–10s)...")
            _FINBERT_PIPELINE = pipeline(
                "sentiment-analysis",
                model=FINBERT_MODEL,
                tokenizer=FINBERT_MODEL,
                truncation=True,
                max_length=512,
            )
            logger.info("FinBERT loaded and ready")
        except Exception as e:
            logger.error(f"Failed to load FinBERT: {e}")
            _FINBERT_PIPELINE = False
    return _FINBERT_PIPELINE if _FINBERT_PIPELINE is not False else None


# ---------------------------------------------------------------------------
# RSS FETCHING
# ---------------------------------------------------------------------------
def _clean_html(text: str) -> str:
    if not text:
        return ""
    return BeautifulSoup(text, 'html.parser').get_text(separator=' ').strip()


def _fetch_yahoo_headlines(ticker: str, max_items: int = 10) -> List[str]:
    """
    Fetch the latest Yahoo Finance RSS headlines for a ticker.
    Returns a list of cleaned 'title. summary' strings, ready for FinBERT.
    """
    url = YAHOO_RSS.format(ticker=ticker)
    try:
        resp = requests.get(
            url, timeout=10,
            headers={'User-Agent': 'Mozilla/5.0 (compatible; DARWIN/1.0)'},
        )
        if resp.status_code != 200:
            logger.warning(f"RSS status {resp.status_code} for {ticker}")
            return []
        feed = feedparser.parse(resp.content)
        headlines = []
        for entry in feed.entries[:max_items]:
            title = _clean_html(entry.get('title', ''))
            summary = _clean_html(entry.get('summary', ''))
            combined = f"{title}. {summary}".strip()
            if combined:
                headlines.append(combined[:512])
        return headlines
    except Exception as e:
        logger.warning(f"RSS fetch failed for {ticker}: {e}")
        return []


# ---------------------------------------------------------------------------
# FINBERT SCORING
# ---------------------------------------------------------------------------
def _score_texts(texts: List[str]) -> List[float]:
    """
    Score each text with FinBERT. Returns a float in [-1, +1] per text.
    Positive → +confidence, Negative → -confidence, Neutral → 0.
    """
    if not texts:
        return []
    nlp = _get_finbert()
    if nlp is None:
        return [0.0] * len(texts)
    try:
        results = nlp(texts, batch_size=8)
    except Exception as e:
        logger.error(f"FinBERT inference failed: {e}")
        return [0.0] * len(texts)

    scores = []
    for r in results:
        label = r['label'].lower()
        conf = float(r['score'])
        if label == 'positive':
            scores.append(+conf)
        elif label == 'negative':
            scores.append(-conf)
        else:
            scores.append(0.0)
    return scores


def get_asset_sentiment(ticker: str, max_items: int = 10
                        ) -> Tuple[float, List[Dict]]:
    """
    Return (aggregate_score, details) for a single ticker.
    aggregate_score is the mean of individual headline scores.
    details is a list of {headline, score} for dashboard display.
    """
    headlines = _fetch_yahoo_headlines(ticker, max_items=max_items)
    if not headlines:
        return 0.0, []
    scores = _score_texts(headlines)
    details = [
        {'headline': h[:140], 'score': round(s, 3)}
        for h, s in zip(headlines, scores)
    ]
    aggregate = float(sum(scores) / len(scores)) if scores else 0.0
    return aggregate, details


# ---------------------------------------------------------------------------
# NEWS AGENT
# ---------------------------------------------------------------------------
class NewsAgent(BaseAgent):
    """
    Sentiment-driven trading agent. Buys on positive aggregate news,
    sells on negative. Adapts thresholds based on period performance.
    """

    def __init__(self, name: str, assets: list, jitter: bool = False,
                 risk_level: float = 5.0):
        super().__init__(name, assets, risk_level=risk_level)
        self.params = {
            'buy_threshold': 0.15,
            'sell_threshold': -0.15,
            'max_headlines': 10,
        }
        # {date_str: {ticker: sentiment_score}} loaded from pickle
        self.sentiment_cache: Dict[str, Dict[str, float]] = {}
        # Last computed scores per asset (for dashboard/debug)
        self.last_scores: Dict[str, float] = {}
        # Cache-of-the-day already fetched (avoid re-fetching within one run)
        self._session_cache: Dict[str, float] = {}

    # ------------------------------------------------------------------
    def load_sentiment_cache(self, path: str) -> None:
        """Load historical {date_str: {ticker: score}} for backtesting."""
        if os.path.exists(path):
            with open(path, 'rb') as f:
                self.sentiment_cache = pickle.load(f)
            logger.info(
                f"Loaded sentiment cache with {len(self.sentiment_cache)} dates"
            )
        else:
            logger.warning(
                f"No sentiment cache at {path}; "
                f"news agent will HOLD during historical backtest."
            )

    # ------------------------------------------------------------------
    def _score_for(self, asset: str, date_str: str) -> float:
        """
        Return sentiment score for (asset, date_str).
        Priority:
          1. Historical cache (backtest)
          2. Session cache (same run)
          3. Live FinBERT fetch — only if date_str is today
          4. Default 0.0 (no data → neutral)
        """
        # 1. Historical cache
        if self.sentiment_cache:
            cached = self.sentiment_cache.get(date_str, {}).get(asset)
            if cached is not None:
                return float(cached)

        # 2. Session cache
        session_key = f"{date_str}:{asset}"
        if session_key in self._session_cache:
            return self._session_cache[session_key]

        # 3. Live fetch — only if this is actually today
        today_str = datetime.now().date().isoformat()
        if date_str == today_str:
            # Global module-level cache for the run
            global_key = f"{today_str}:{asset}"
            if global_key in _SENTIMENT_CACHE:
                score, _ = _SENTIMENT_CACHE[global_key]
            else:
                score, details = get_asset_sentiment(
                    asset, self.params['max_headlines']
                )
                _SENTIMENT_CACHE[global_key] = (score, details)
            self._session_cache[session_key] = score
            return score

        # 4. No data → neutral
        return 0.0

    # ------------------------------------------------------------------
    def generate_signals(
        self, market_data: pd.DataFrame, current_prices: pd.Series
    ) -> Dict[str, Signal]:
        signals: Dict[str, Signal] = {}

        # Current date from the market data index
        try:
            current_date = market_data.index[-1]
            date_str = str(current_date.date())
        except Exception:
            return {a: 'HOLD' for a in self.assets}

        # Scale thresholds by agent risk
        eff_buy = self.params['buy_threshold'] / self.signal_aggression
        eff_sell = self.params['sell_threshold'] / self.signal_aggression

        for asset in self.assets:
            score = self._score_for(asset, date_str)
            self.last_scores[asset] = score

            if score > eff_buy:
                signals[asset] = 'BUY'
            elif score < eff_sell:
                signals[asset] = 'SELL'
            else:
                signals[asset] = 'HOLD'

        return signals

    # ------------------------------------------------------------------
    def adapt_parameters(self, performance_metrics: Dict,
                         mutation_info: Dict = None) -> None:
        """
        Tighten thresholds on loss, loosen on win. Honors mutation_info
        from the teammate's mutation system.
        """
        ret = performance_metrics.get('return', 0.0)

        if ret < 0:
            # Wait for stronger signals
            self.params['buy_threshold'] *= 1.10
            self.params['sell_threshold'] *= 1.10
        else:
            # Trust the signal more
            self.params['buy_threshold'] *= 0.95
            self.params['sell_threshold'] *= 0.95

        # Clamp
        self.params['buy_threshold'] = max(
            0.05, min(0.5, self.params['buy_threshold'])
        )
        self.params['sell_threshold'] = max(
            -0.5, min(-0.05, self.params['sell_threshold'])
        )

        # Honor mutation system suggestions if provided
        if mutation_info:
            for key, value in mutation_info.items():
                if key in self.params:
                    self.params[key] = value