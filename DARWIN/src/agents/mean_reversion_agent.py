"""
Mean reversion agent: buys oversold dips, sells overbought peaks.

RISK SCALING:
  The effective z-score threshold is divided by signal_aggression.
  High-risk agents fire at lower z-scores (easier to trigger);
  low-risk agents wait for more extreme deviations.
"""

from typing import Dict

import pandas as pd

from src.agents.base_agent import BaseAgent, Signal


class MeanReversionAgent(BaseAgent):

    def __init__(self, name: str, assets: list, jitter: bool = False,
                 risk_level: float = 5.0):
        super().__init__(name, assets, risk_level=risk_level)
        import random
        j = (lambda lo, hi: random.uniform(lo, hi)) if jitter else \
            (lambda lo, hi: (lo + hi) / 2)
        self.params = {
            'lookback': int(j(15, 30)),
            'z_threshold': j(1.5, 2.5),
            'exit_z': 0.5,
        }

    def generate_signals(
        self, market_data: pd.DataFrame, current_prices: pd.Series
    ) -> Dict[str, Signal]:
        signals: Dict[str, Signal] = {}
        close = market_data.xs('Close', axis=1, level=1)
        lb = self.params['lookback']
        ez = self.params['exit_z']

        # High risk -> lower effective z -> more entries
        eff_z = self.params['z_threshold'] / self.signal_aggression

        if len(close) < lb + 1:
            return {a: 'HOLD' for a in self.assets}

        for asset in self.assets:
            if asset not in close.columns:
                signals[asset] = 'HOLD'
                continue
            prices = close[asset].dropna()
            if len(prices) < lb + 1:
                signals[asset] = 'HOLD'
                continue

            window = prices.iloc[-lb:]
            mean = window.mean()
            std = window.std()
            if std == 0 or pd.isna(std):
                signals[asset] = 'HOLD'
                continue
            z = (prices.iloc[-1] - mean) / std

            if z < -eff_z:
                signals[asset] = 'BUY'
            elif z > eff_z:
                signals[asset] = 'SELL'
            elif abs(z) < ez:
                signals[asset] = 'SELL'
            else:
                signals[asset] = 'HOLD'
        return signals

    def adapt_parameters(self, performance_metrics: Dict,
                         mutation_info: Dict = None) -> None:
        """
        Two-layer adaptation:
          1. Mild z_threshold tweak based on sign of return (existing logic).
             - Losing → widen threshold (be pickier about entries).
             - Winning → tighten threshold (allow more trades).
          2. Strong parameter mutation when flagged as an underperformer
             by the reallocation layer.
        """
        # ---- Layer 1: existing mild z_threshold tweak ----
        ret = performance_metrics.get('return', 0.0)
        if ret < 0:
            self.params['z_threshold'] *= 1.1
        else:
            self.params['z_threshold'] *= 0.95
        self.params['z_threshold'] = max(
            1.0, min(3.5, self.params['z_threshold'])
        )

        # ---- Layer 2: strong mutation if flagged ----
        if mutation_info and mutation_info.get('needs_mutation'):
            strength = mutation_info.get('strength', 0.1)
            self.mutate(strength=strength)