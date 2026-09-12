"""
Contrarian agent: bets against recent extremes.

RISK SCALING:
  The effective sentiment threshold is divided by signal_aggression.
  High-risk agents act on smaller moves; low-risk agents wait for
  bigger dislocations before fading the crowd.
"""

from typing import Dict

import pandas as pd

from src.agents.base_agent import BaseAgent, Signal


class ContrarianAgent(BaseAgent):

    def __init__(self, name: str, assets: list, jitter: bool = False,
                 risk_level: float = 5.0):
        super().__init__(name, assets, risk_level=risk_level)
        import random
        j = (lambda lo, hi: random.uniform(lo, hi)) if jitter else \
            (lambda lo, hi: (lo + hi) / 2)
        self.params = {
            'sentiment_window': int(j(10, 30)),
            'reversal_strength': j(0.7, 1.3),
        }

    def generate_signals(
        self, market_data: pd.DataFrame, current_prices: pd.Series
    ) -> Dict[str, Signal]:
        signals: Dict[str, Signal] = {}
        close = market_data.xs('Close', axis=1, level=1)
        w = self.params['sentiment_window']

        # High risk -> smaller effective threshold -> more contrarian bets
        eff_rs = self.params['reversal_strength'] / self.signal_aggression

        if len(close) < w + 1:
            return {a: 'HOLD' for a in self.assets}

        for asset in self.assets:
            if asset not in close.columns:
                signals[asset] = 'HOLD'
                continue
            prices = close[asset].dropna()
            if len(prices) < w + 1:
                signals[asset] = 'HOLD'
                continue

            sentiment = (prices.iloc[-1] / prices.iloc[-w]) - 1

            if sentiment < -0.05 * eff_rs:
                signals[asset] = 'BUY'
            elif sentiment > 0.05 * eff_rs:
                signals[asset] = 'SELL'
            else:
                signals[asset] = 'HOLD'
        return signals

    def adapt_parameters(self, performance_metrics: Dict,
                         mutation_info: Dict = None) -> None:
        """
        Two-layer adaptation:
          1. Mild reversal_strength tweak based on sign of return (existing).
             - Losing  → raise threshold (only fade bigger extremes).
             - Winning → lower threshold (act on smaller dislocations).
          2. Strong parameter mutation when flagged as an underperformer
             by the reallocation layer.
        """
        # ---- Layer 1: existing reversal_strength tweak ----
        ret = performance_metrics.get('return', 0.0)
        if ret < 0:
            self.params['reversal_strength'] *= 1.1
        else:
            self.params['reversal_strength'] *= 0.95
        self.params['reversal_strength'] = max(
            0.5, min(2.5, self.params['reversal_strength'])
        )

        # ---- Layer 2: strong mutation if flagged ----
        if mutation_info and mutation_info.get('needs_mutation'):
            strength = mutation_info.get('strength', 0.1)
            self.mutate(strength=strength)