"""
ML agent: rule-based scoring using momentum, volatility, and volume.

No live model training — too slow for a 30-second backtest. Instead we
compute a weighted z-score of three features and let the weights drift
based on which feature best predicted the next day's return.

RISK SCALING:
  The score_threshold is divided by signal_aggression. High-risk agents
  trade on weaker scores, low-risk agents demand stronger conviction.
"""

from collections import deque
from typing import Deque, Dict, List, Tuple

import numpy as np
import pandas as pd

from src.agents.base_agent import BaseAgent, Signal
from src.utils import config


class MLAgent(BaseAgent):

    def __init__(self, name: str, assets: list, jitter: bool = False,
                 risk_level: float = 5.0):
        super().__init__(name, assets, risk_level=risk_level)
        import random
        j = (lambda lo, hi: random.uniform(lo, hi)) if jitter else \
            (lambda lo, hi: (lo + hi) / 2)
        self.params = {
            'lookback': int(j(5, 15)),
            'w_momentum': j(0.3, 0.5),
            'w_volatility': -j(0.2, 0.4),
            'w_volume': j(0.2, 0.4),
            'score_threshold': 0.1,
        }
        self.feature_history: Deque[Tuple[float, float, float, float]] = deque(
            maxlen=config.DEFAULTS['ml_history_window']
        )
        self._pending_features: Dict[str, Tuple[float, float, float]] = {}

    # ------------------------------------------------------------------
    def _compute_features(self, close: pd.DataFrame, volume: pd.DataFrame,
                          i: int) -> Dict[str, Tuple[float, float, float]]:
        lb = self.params['lookback']
        feats = {}
        for asset in self.assets:
            if asset not in close.columns or i < lb:
                continue
            prices = close[asset].iloc[:i + 1]
            vols = volume[asset].iloc[:i + 1] if asset in volume.columns else None

            momentum = (prices.iloc[-1] / prices.iloc[-lb]) - 1

            window = prices.iloc[-lb:]
            mean = window.mean()
            vol = window.std() / mean if mean > 0 else 0.0

            if vols is not None and len(vols) >= lb:
                vmean = vols.iloc[-lb:].mean()
                v = (vols.iloc[-1] / vmean - 1) if vmean > 0 else 0.0
            else:
                v = 0.0

            feats[asset] = (momentum, vol, v)
        return feats

    def _cross_sectional_z(self, values: Dict[str, float]) -> Dict[str, float]:
        if len(values) < 2:
            return {k: 0.0 for k in values}
        arr = np.array(list(values.values()))
        mean = arr.mean()
        std = arr.std()
        if std == 0:
            return {k: 0.0 for k in values}
        return {k: (v - mean) / std for k, v in values.items()}

    # ------------------------------------------------------------------
    def generate_signals(
        self, market_data: pd.DataFrame, current_prices: pd.Series
    ) -> Dict[str, Signal]:
        signals: Dict[str, Signal] = {}
        close = market_data.xs('Close', axis=1, level=1)
        volume = market_data.xs('Volume', axis=1, level=1)
        i = len(close) - 1

        # --- backfill next-day return for yesterday's features ---
        if i > 0 and self._pending_features:
            feats_yest = self._pending_features
            next_rets = {}
            for a, _ in feats_yest.items():
                if a in close.columns:
                    p_today = close[a].iloc[i]
                    p_yest = close[a].iloc[i - 1]
                    if p_yest > 0:
                        next_rets[a] = (p_today / p_yest) - 1
            if next_rets:
                self.feature_history.append((
                    float(np.mean([f[0] for f in feats_yest.values()])),
                    float(np.mean([f[1] for f in feats_yest.values()])),
                    float(np.mean([f[2] for f in feats_yest.values()])),
                    float(np.mean(list(next_rets.values()))),
                ))
            self._pending_features = {}

        # --- compute today's features ---
        feats = self._compute_features(close, volume, i)
        if not feats:
            return {a: 'HOLD' for a in self.assets}

        self._pending_features = feats

        # --- z-score each feature cross-sectionally ---
        mom_z = self._cross_sectional_z({a: feats[a][0] for a in feats})
        vol_z = self._cross_sectional_z({a: feats[a][1] for a in feats})
        volume_z = self._cross_sectional_z({a: feats[a][2] for a in feats})

        wm = self.params['w_momentum']
        wv = self.params['w_volatility']
        wvol = self.params['w_volume']
        # High risk -> lower effective threshold -> more trades
        eff_thr = self.params['score_threshold'] / self.signal_aggression

        for asset in self.assets:
            if asset not in feats:
                signals[asset] = 'HOLD'
                continue
            score = (wm * mom_z[asset]
                     + wv * vol_z[asset]
                     + wvol * volume_z[asset])
            if score > eff_thr:
                signals[asset] = 'BUY'
            elif score < -eff_thr:
                signals[asset] = 'SELL'
            else:
                signals[asset] = 'HOLD'
        return signals

    # ------------------------------------------------------------------
    def adapt_parameters(self, performance_metrics: Dict,
                         mutation_info: Dict = None) -> None:
        """
        Two-layer adaptation:
          1. Correlation-based feature reweighting (existing logic).
             Uses feature_history to find which of momentum / volatility /
             volume best predicted next-day returns, and shifts weights.
          2. Strong parameter mutation when flagged as an underperformer
             by the reallocation layer.
        """
        # ---- Layer 1: correlation-based reweighting (existing) ----
        if len(self.feature_history) >= 10:
            arr = np.array(self.feature_history)
            feats = arr[:, :3]
            next_rets = arr[:, 3]

            if np.std(next_rets) != 0:
                corrs = []
                for j in range(3):
                    if np.std(feats[:, j]) == 0:
                        corrs.append(0.0)
                    else:
                        corrs.append(
                            float(np.corrcoef(feats[:, j], next_rets)[0, 1])
                        )
                corrs = np.nan_to_num(np.array(corrs))

                best = int(np.argmax(np.abs(corrs)))
                worst = int(np.argmin(np.abs(corrs)))

                weights = [self.params['w_momentum'],
                           self.params['w_volatility'],
                           self.params['w_volume']]

                weights[best] += 0.1 * np.sign(corrs[best])
                weights[worst] -= (
                    0.1 * np.sign(corrs[worst]) if corrs[worst] != 0 else 0.0
                )

                if weights[1] > 0:
                    weights[1] = -abs(weights[1])

                total = sum(abs(w) for w in weights)
                if total > 0:
                    weights = [w / total for w in weights]

                self.params['w_momentum'] = weights[0]
                self.params['w_volatility'] = weights[1]
                self.params['w_volume'] = weights[2]

        # ---- Layer 2: strong mutation if flagged ----
        if mutation_info and mutation_info.get('needs_mutation'):
            strength = mutation_info.get('strength', 0.1)
            self.mutate(strength=strength)