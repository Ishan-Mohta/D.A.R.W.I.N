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
    def adapt_parameters(self, performance_metrics: Dict) -> None:
        """Mutate self.params based on period performance."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------
    def get_params(self) -> Dict:
        return dict(self.params)

    def __repr__(self) -> str:
        return (f"<{self.__class__.__name__} name={self.name} "
                f"risk={self.risk_level:.2f}>")