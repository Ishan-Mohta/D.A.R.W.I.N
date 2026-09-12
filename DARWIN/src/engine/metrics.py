"""Pure metric functions. No state, easy to unit test."""

from typing import Dict, List

import numpy as np
import pandas as pd


def calculate_return(pv_list: List[float], initial: float) -> float:
    if not pv_list or initial <= 0:
        return 0.0
    return (pv_list[-1] - initial) / initial


def calculate_sharpe(daily_returns: List[float],
                     rf: float = 0.0, periods: int = 252) -> float:
    if len(daily_returns) < 2:
        return 0.0
    r = np.array(daily_returns)
    excess = r - rf / periods
    std = excess.std()
    if std == 0:
        return 0.0
    return float(np.sqrt(periods) * excess.mean() / std)


def calculate_max_drawdown(pv_list: List[float]) -> float:
    if not pv_list:
        return 0.0
    pv = np.array(pv_list)
    peak = np.maximum.accumulate(pv)
    dd = (pv - peak) / peak
    return float(dd.min())


def calculate_win_rate(trades: List[Dict]) -> float:
    """Round-trip win rate per asset."""
    if not trades:
        return 0.0
    buys: Dict[str, List[float]] = {}
    wins = losses = 0
    for t in trades:
        if t['side'] == 'BUY' and t['filled_shares'] > 0:
            buys.setdefault(t['asset'], []).append(t['price'])
        elif t['side'] == 'SELL' and t['filled_shares'] > 0:
            if buys.get(t['asset']):
                avg_buy = sum(buys[t['asset']]) / len(buys[t['asset']])
                if t['price'] > avg_buy:
                    wins += 1
                else:
                    losses += 1
    total = wins + losses
    return wins / total if total > 0 else 0.0


def calculate_benchmark_return(market_data: pd.DataFrame,
                               ticker: str) -> float:
    """Buy-and-hold return of the benchmark over the period."""
    try:
        close = market_data.xs('Close', axis=1, level=1)[ticker].dropna()
        if len(close) < 2:
            return 0.0
        return float((close.iloc[-1] / close.iloc[0]) - 1)
    except Exception:
        return 0.0