"""
CentralBroker — the single source of truth for all capital and positions.

Every agent submits orders here. The broker validates, caps, fills, and
logs. Guardrails (position size, cash buffer, drawdown kill) are
enforced in exactly one place.

RISK SCALING:
  Each agent carries its own max_position_size and min_cash_buffer
  derived from its sampled risk_level. The broker looks these up
  per-agent when executing orders, so a risk=9 bot can hold 30%+ in a
  single position while a risk=1 bot is capped near 8%.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List

import pandas as pd

from src.engine.execution import validate_and_cap_buy
from src.engine.metrics import (
    calculate_max_drawdown,
    calculate_return,
    calculate_sharpe,
    calculate_win_rate,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Order:
    timestamp: pd.Timestamp
    agent_id: str
    asset: str
    side: str
    requested_value: float
    filled_shares: int
    filled_value: float
    price: float
    status: str
    reason: str


@dataclass
class SubAccount:
    agent_id: str
    cash: float
    positions: Dict[str, int] = field(default_factory=dict)
    portfolio_history: List[float] = field(default_factory=list)
    returns: List[float] = field(default_factory=list)
    period_start_capital: float = 0.0
    period_start_value: float = 0.0
    killed: bool = False


class CentralBroker:

    def __init__(self, agent_ids: List[str], initial_capital: float,
                 assets: List[str], config: dict,
                 risk_params: Dict[str, Dict] = None):
        self.agent_ids = list(agent_ids)
        self.assets = list(assets)
        self.config = config
        # Per-agent risk parameters: {agent_id: {max_position_size, min_cash_buffer}}
        self._risk_params: Dict[str, Dict] = risk_params or {}

        n = len(agent_ids)
        per_agent = initial_capital / n

        self.accounts: Dict[str, SubAccount] = {
            aid: SubAccount(
                agent_id=aid,
                cash=per_agent,
                positions={a: 0 for a in assets},
                period_start_capital=per_agent,
                period_start_value=per_agent,
            )
            for aid in agent_ids
        }
        self.trade_log: List[Order] = []
        self.capital_history: List[Dict] = []

    # ------------------------------------------------------------------
    # RISK LOOKUP
    # ------------------------------------------------------------------
    def _get_risk(self, agent_id: str, key: str, fallback: float) -> float:
        """Fetch a per-agent risk parameter, falling back to global config."""
        rp = self._risk_params.get(agent_id)
        if rp and key in rp:
            return float(rp[key])
        return float(self.config.get(key, fallback))

    # ------------------------------------------------------------------
    # ACCESS
    # ------------------------------------------------------------------
    def get_account(self, agent_id: str) -> SubAccount:
        return self.accounts[agent_id]

    def get_portfolio_value(self, agent_id: str, prices: pd.Series) -> float:
        acct = self.accounts[agent_id]
        value = acct.cash
        for asset, shares in acct.positions.items():
            if shares > 0 and asset in prices.index:
                price = prices[asset]
                if pd.notna(price):
                    value += shares * float(price)
        return value

    def get_total_value(self, prices: pd.Series) -> float:
        return sum(self.get_portfolio_value(aid, prices)
                   for aid in self.agent_ids)

    # ------------------------------------------------------------------
    # ORDER EXECUTION
    # ------------------------------------------------------------------
    def execute_order(self, agent_id: str, asset: str, side: str,
                      desired_value: float, price: float,
                      timestamp: pd.Timestamp) -> Order:
        acct = self.accounts[agent_id]

        if acct.killed:
            order = Order(timestamp, agent_id, asset, side, desired_value,
                          0, 0.0, price, 'REJECTED', 'agent_killed')
            self.trade_log.append(order)
            return order

        if pd.isna(price) or price <= 0:
            order = Order(timestamp, agent_id, asset, side, desired_value,
                          0, 0.0, price, 'REJECTED', 'bad_price')
            self.trade_log.append(order)
            return order

        price = float(price)

        if side == 'BUY':
            return self._execute_buy(agent_id, asset, desired_value,
                                     price, timestamp)
        elif side == 'SELL':
            return self._execute_sell(agent_id, asset, desired_value,
                                      price, timestamp)
        else:
            raise ValueError(f"Unknown side: {side}")

    def _execute_buy(self, agent_id: str, asset: str, desired_value: float,
                     price: float, timestamp: pd.Timestamp) -> Order:
        acct = self.accounts[agent_id]
        prices_series = pd.Series([price], index=[asset], dtype='float64')
        portfolio_value = self.get_portfolio_value(agent_id, prices_series)
        existing = acct.positions.get(asset, 0) * price

        # ---- Per-agent risk parameters ----
        max_pos = self._get_risk(agent_id, 'max_position_size', 0.15)
        min_cash = self._get_risk(agent_id, 'min_cash_buffer', 0.02)

        capped, reason = validate_and_cap_buy(
            desired_value=desired_value,
            agent_portfolio_value=portfolio_value,
            existing_position_value=existing,
            agent_cash=acct.cash,
            max_position_size=max_pos,
            min_cash_buffer=min_cash,
        )

        if capped <= 0:
            order = Order(timestamp, agent_id, asset, 'BUY', desired_value,
                          0, 0.0, price, 'REJECTED', reason)
            self.trade_log.append(order)
            return order

        shares = int(capped // price)
        if shares <= 0:
            order = Order(timestamp, agent_id, asset, 'BUY', desired_value,
                          0, 0.0, price, 'REJECTED', 'below_one_share')
            self.trade_log.append(order)
            return order

        filled_value = shares * price
        acct.cash -= filled_value
        acct.positions[asset] = acct.positions.get(asset, 0) + shares

        status = 'FILLED' if capped >= desired_value - 1e-6 else 'CAPPED'
        order = Order(timestamp, agent_id, asset, 'BUY', desired_value,
                      shares, filled_value, price, status, reason)
        self.trade_log.append(order)
        return order

    def _execute_sell(self, agent_id: str, asset: str, desired_value: float,
                      price: float, timestamp: pd.Timestamp) -> Order:
        acct = self.accounts[agent_id]
        shares_held = acct.positions.get(asset, 0)
        if shares_held <= 0:
            order = Order(timestamp, agent_id, asset, 'SELL', desired_value,
                          0, 0.0, price, 'REJECTED', 'insufficient_shares')
            self.trade_log.append(order)
            return order

        if desired_value > 0:
            desired_shares = int(desired_value // price)
            shares = min(shares_held, desired_shares) if desired_shares > 0 \
                else shares_held
        else:
            shares = shares_held

        proceeds = shares * price
        acct.cash += proceeds
        acct.positions[asset] = shares_held - shares

        order = Order(timestamp, agent_id, asset, 'SELL', desired_value,
                      shares, proceeds, price, 'FILLED', 'ok')
        self.trade_log.append(order)
        return order

    # ------------------------------------------------------------------
    # MARK TO MARKET
    # ------------------------------------------------------------------
    def mark_to_market(self, prices: pd.Series,
                       timestamp: pd.Timestamp) -> None:
        for aid in self.agent_ids:
            acct = self.accounts[aid]
            pv = self.get_portfolio_value(aid, prices)
            if acct.portfolio_history:
                prev = acct.portfolio_history[-1]
                if prev > 0:
                    acct.returns.append((pv - prev) / prev)
            acct.portfolio_history.append(pv)

    # ------------------------------------------------------------------
    # REDISTRIBUTION
    # ------------------------------------------------------------------
    def redistribute_capital(self, new_allocations: Dict[str, float],
                             prices: pd.Series,
                             timestamp: pd.Timestamp) -> None:
        """
        Liquidate at `prices`, reset each agent's cash to its new
        allocation, clear positions, and record a snapshot.
        """
        snapshot = {'timestamp': str(timestamp.date()), 'agents': {}}

        for aid in self.agent_ids:
            acct = self.accounts[aid]
            old_cap = acct.period_start_capital
            end_value = self.get_portfolio_value(aid, prices)
            period_return = ((end_value - old_cap) / old_cap
                             if old_cap > 0 else 0.0)
            profit = end_value - old_cap
            new_cap = new_allocations[aid]

            snapshot['agents'][aid] = {
                'old_cap': round(old_cap, 2),
                'new_cap': round(new_cap, 2),
                'return': round(period_return, 4),
                'profit': round(profit, 2),
            }

            acct.cash = new_cap
            acct.positions = {a: 0 for a in self.assets}
            acct.portfolio_history = []
            acct.returns = []
            acct.period_start_capital = new_cap
            acct.period_start_value = new_cap
            acct.killed = False

        self.capital_history.append(snapshot)

    # ------------------------------------------------------------------
    # METRICS
    # ------------------------------------------------------------------
    def get_all_metrics(self) -> Dict[str, Dict]:
        out = {}
        for aid in self.agent_ids:
            acct = self.accounts[aid]
            trades = [asdict(o) for o in self.trade_log if o.agent_id == aid]
            out[aid] = {
                'name': aid,
                'capital_at_start': acct.period_start_capital,
                'return': calculate_return(
                    acct.portfolio_history, acct.period_start_capital
                ),
                'sharpe': calculate_sharpe(acct.returns),
                'max_drawdown': calculate_max_drawdown(acct.portfolio_history),
                'win_rate': calculate_win_rate(trades),
                'num_trades': len([t for t in trades
                                   if t['status'] in ('FILLED', 'CAPPED')]),
            }
        return out

    def rank_agents(self) -> List[str]:
        metrics = self.get_all_metrics()
        return sorted(self.agent_ids,
                      key=lambda a: metrics[a]['return'],
                      reverse=True)