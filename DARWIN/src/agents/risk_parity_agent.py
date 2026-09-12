"""
Risk parity agent: equalizes risk contribution across assets by
inverse-volatility weighting.

RISK SCALING:
  The rebalance_threshold is divided by signal_aggression. High-risk
  agents rebalance more often (smaller gaps trigger), low-risk agents
  let positions drift further before acting.

  Also, the agent's target weight concentrates more in volatile assets
  at high risk levels (higher risk -> more willing to hold volatile
  names for bigger potential returns).
"""

from typing import Dict

import pandas as pd

from src.agents.base_agent import BaseAgent, Signal


class RiskParityAgent(BaseAgent):

    def __init__(self, name: str, assets: list, jitter: bool = False,
                 risk_level: float = 5.0):
        super().__init__(name, assets, risk_level=risk_level)
        import random
        j = (lambda lo, hi: random.uniform(lo, hi)) if jitter else \
            (lambda lo, hi: (lo + hi) / 2)
        self.params = {
            'vol_lookback': int(j(15, 30)),
            'rebalance_threshold': j(0.05, 0.15),
        }

    def generate_signals(
        self, market_data: pd.DataFrame, current_prices: pd.Series
    ) -> Dict[str, Signal]:
        signals: Dict[str, Signal] = {}
        close = market_data.xs('Close', axis=1, level=1)
        lb = self.params['vol_lookback']

        # High risk -> smaller effective threshold -> more rebalances
        eff_thr = self.params['rebalance_threshold'] / self.signal_aggression

        if len(close) < lb + 1:
            return {a: 'HOLD' for a in self.assets}

        # --- compute inverse-vol weights ---
        inv_vols = {}
        for asset in self.assets:
            if asset not in close.columns:
                continue
            prices = close[asset].dropna()
            if len(prices) < lb:
                continue
            vol = prices.iloc[-lb:].pct_change().std()
            if vol > 0:
                inv_vols[asset] = 1.0 / vol

        if not inv_vols:
            return {a: 'HOLD' for a in self.assets}

        # --- risk tilt: at higher risk, tilt weights toward higher-vol assets ---
        # We re-weight: w_i' = w_i^alpha where alpha = 1.0 (risk 0) -> -0.5 (risk 10)
        # alpha < 1 favours higher-vol assets
        alpha = 1.0 - self.risk_0_to_1 * 1.5
        tilted = {a: (v ** alpha) for a, v in inv_vols.items()}

        total = sum(tilted.values())
        if total <= 0:
            return {a: 'HOLD' for a in self.assets}
        weights = {a: v / total for a, v in tilted.items()}

        # --- compare weights to equal-weight to decide signals ---
        n = len(weights)
        equal = 1.0 / n

        for asset in self.assets:
            if asset not in weights:
                signals[asset] = 'HOLD'
                continue
            gap = weights[asset] - equal
            if gap > eff_thr:
                signals[asset] = 'BUY'
            elif gap < -eff_thr:
                signals[asset] = 'SELL'
            else:
                signals[asset] = 'HOLD'
        return signals

    def adapt_parameters(self, performance_metrics: Dict) -> None:
        ret = performance_metrics.get('return', 0.0)
        if ret < 0:
            self.params['vol_lookback'] = min(
                60, self.params['vol_lookback'] + 5
            )
        else:
            self.params['vol_lookback'] = max(
                10, self.params['vol_lookback'] - 2
            )