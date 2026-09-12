"""
Abstract BaseAgent.

CRITICAL DESIGN: agents do NOT own cash or positions. They are pure
signal generators. The CentralBroker owns all state.

RISK APPETITE:
  Each agent carries its own risk_level (0-10). Derived properties
  (max_position_size, min_cash_buffer, signal_aggression) scale smoothly
  from "ultra-conservative" at 0 to "degen" at 10. The registry samples
  risk_level from a Gaussian centered on the user's choice.
"""

import random
from abc import ABC, abstractmethod
from typing import Dict, List, Union

import pandas as pd

Signal = Union[str, tuple]   # 'BUY' | 'SELL' | 'HOLD' | ('BUY', dollar_amt)


class BaseAgent(ABC):
    """Every trading strategy inherits from this."""

    def __init__(self, name: str, assets: List[str], risk_level: float = 5.0):
        self.name = name
        self.assets = list(assets)
        self.params: Dict = {}
        self.risk_level = float(risk_level)

    # ------------------------------------------------------------------
    # RISK-DERIVED PROPERTIES
    # ------------------------------------------------------------------
    @property
    def risk_0_to_1(self) -> float:
        """Normalized risk in [0, 1]."""
        return max(0.0, min(1.0, self.risk_level / 10.0))

    @property
    def max_position_size(self) -> float:
        """Max % of portfolio in a single asset. 5% at risk 0, 35% at risk 10."""
        return 0.05 + self.risk_0_to_1 * 0.30

    @property
    def min_cash_buffer(self) -> float:
        """Minimum cash reserve. 15% at risk 0, 2% at risk 10."""
        return 0.15 - self.risk_0_to_1 * 0.13

    @property
    def signal_aggression(self) -> float:
        """
        Multiplier on signal sensitivity. High-risk agents act on weaker
        signals (thresholds divided by this value). 0.5 at risk 0, 2.0
        at risk 10.
        """
        return 0.5 + self.risk_0_to_1 * 1.5

    # ------------------------------------------------------------------
    # ABSTRACT INTERFACE
    # ------------------------------------------------------------------
    @abstractmethod
    def generate_signals(
        self,
        market_data: pd.DataFrame,
        current_prices: pd.Series,
    ) -> Dict[str, Signal]:
        """
        Return {asset: 'BUY' | 'SELL' | 'HOLD'} or
               {asset: ('BUY', desired_dollar_amount)}.

        `market_data` MUST ONLY contain rows up to and including today.
        """
        raise NotImplementedError

    @abstractmethod
    def adapt_parameters(self, performance_metrics: Dict,
                         mutation_info: Dict = None) -> None:
        """
        Mutate self.params based on period performance.

        mutation_info: dict from reallocation.py with keys
                       'needs_mutation' (bool) and 'strength' (float).
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # MUTATION — used by subclasses when flagged as underperformers
    # ------------------------------------------------------------------
    def mutate(self, strength: float = 0.2) -> Dict:
        """
        Perturb parameters to explore new behavior. Called when the agent
        has been flagged as an underperformer.

        strength: 0.0 = no change, 1.0 = maximum chaos
        Returns: dict of {param_name: (old_value, new_value)} for logging.
        """
        changes = {}
        strength = max(0.0, min(1.0, strength))

        # --- 1. Risk level (the big lever) ---
        # On mutation, a loser can swing risk dramatically in either
        # direction. If losing badly, we're equally willing to try
        # "much safer" or "much more aggressive" — evolution doesn't
        # have a bias toward either.
        old_risk = self.risk_level
        delta_risk = random.gauss(0.0, strength * 3.0)  # up to ±3 at max
        self.risk_level = max(0.5, min(10.0, old_risk + delta_risk))
        if abs(self.risk_level - old_risk) > 1e-6:
            changes['risk_level'] = (old_risk, self.risk_level)

        # --- 2. Numeric params in self.params ---
        # Every subclass stores its strategy-specific tunables here.
        # We perturb each one proportionally.
        for key, old_val in list(self.params.items()):
            if not isinstance(old_val, (int, float)):
                continue  # skip non-numeric params

            # Perturbation magnitude scales with the size of the param
            # so that big numbers aren't mutated into oblivion.
            scale = max(abs(old_val), 0.1)
            delta = random.gauss(0.0, strength * scale * 0.5)
            new_val = old_val + delta

            # Keep signs sane (don't flip lookback from +10 to -10)
            if old_val >= 0:
                new_val = max(0.0, new_val)
            if isinstance(old_val, int):
                new_val = int(round(new_val))

            self.params[key] = new_val
            changes[key] = (old_val, new_val)

        return changes

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------
    def get_params(self) -> Dict:
        """All tunable parameters, including risk_level, for logging."""
        p = dict(self.params)
        p['risk_level'] = self.risk_level
        return p

    def __repr__(self) -> str:
        return (f"<{self.__class__.__name__} name={self.name} "
                f"risk={self.risk_level:.2f}>")