"""
Momentum agent: buys uptrends, sells downtrends.

RISK SCALING:
  The effective sensitivity is divided by signal_aggression. A high-risk
  agent (aggression ~2.0) fires on half the signal strength a low-risk
  agent (aggression ~0.5) would need. Net effect: aggressive bots trade
  more often and with bigger positions.
"""

from typing import Dict

import pandas as pd

from src.agents.base_agent import BaseAgent, Signal


class MomentumAgent(BaseAgent):

    def __init__(self, name: str, assets: list, jitter: bool = False,
                 risk_level: float = 5.0):
        super().__init__(name, assets, risk_level=risk_level)
        import random
        j = (lambda lo, hi: random.uniform(lo, hi)) if jitter else \
            (lambda lo, hi: (lo + hi) / 2)
        self.params = {
            'short_window': int(j(3, 8)),
            'long_window': int(j(15, 30)),
            'sensitivity': j(0.7, 1.3),
        }

    def generate_signals(
        self, market_data: pd.DataFrame, current_prices: pd.Series
    ) -> Dict[str, Signal]:
        signals: Dict[str, Signal] = {}
        close = market_data.xs('Close', axis=1, level=1)
        s = self.params['short_window']
        l = self.params['long_window']

        # Effective sensitivity: high risk -> lower threshold -> more triggers
        eff_sens = self.params['sensitivity'] / self.signal_aggression

        if len(close) < l + 1:
            return {a: 'HOLD' for a in self.assets}

        for asset in self.assets:
            if asset not in close.columns:
                signals[asset] = 'HOLD'
                continue
            prices = close[asset].dropna()
            if len(prices) < l + 1:
                signals[asset] = 'HOLD'
                continue

            short_ret = (prices.iloc[-1] / prices.iloc[-s]) - 1
            long_ret = (prices.iloc[-1] / prices.iloc[-l]) - 1

            if short_ret > long_ret * eff_sens:
                signals[asset] = 'BUY'
            elif short_ret < -long_ret * eff_sens:
                signals[asset] = 'SELL'
            else:
                signals[asset] = 'HOLD'
        return signals

    def adapt_parameters(self, performance_metrics: Dict) -> None:
        ret = performance_metrics.get('return', 0.0)
        if ret < 0:
            self.params['sensitivity'] *= 0.9
        else:
            self.params['sensitivity'] *= 1.05
        self.params['sensitivity'] = max(
            0.3, min(2.0, self.params['sensitivity'])
        )