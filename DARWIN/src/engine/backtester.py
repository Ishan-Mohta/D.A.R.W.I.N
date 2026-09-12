"""
Backtester — the main simulation loop.

Walks through every trading day:
  1. each agent generates signals from data up to today
  2. the broker executes the resulting orders with guardrails
  3. mark-to-market updates everyone's value
  4. on rebalance days, capital is redistributed and params adapt

Zero look-ahead bias: agents only see market_data.iloc[:i+1].
"""

from dataclasses import asdict
from typing import Dict, List

import pandas as pd

from src.adaptation.reallocation import compute_new_allocations
from src.engine.broker import CentralBroker
from src.engine.metrics import calculate_benchmark_return
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Backtester:

    def __init__(self, agents: list, market_data: pd.DataFrame,
                 broker: CentralBroker, config: dict):
        self.agents = agents
        self.market_data = market_data
        self.broker = broker
        self.config = config

    # ------------------------------------------------------------------
    def _should_rebalance(self, current: pd.Timestamp,
                          last: pd.Timestamp) -> bool:
        """Return True if we've crossed into a new W/M/Q period."""
        freq = self.config['rebalance_freq']
        if freq == 'W':
            return (current.isocalendar().week != last.isocalendar().week
                    or current.year != last.year)
        if freq == 'M':
            return current.month != last.month or current.year != last.year
        if freq == 'Q':
            return current.quarter != last.quarter or current.year != last.year
        return False

    # ------------------------------------------------------------------
    def run(self) -> Dict:
        """Execute the full backtest and return a results dict."""
        close = self.market_data.xs('Close', axis=1, level=1)
        trading_days = close.index
        assets = self.config['assets']

        last_rebalance = trading_days[0]
        daily_pool_values: List[float] = []
        param_changes: List[Dict] = []

        for i, date in enumerate(trading_days):
            prices = close.iloc[i]

            # ---- 1. agents generate signals, broker executes ----
            for agent in self.agents:
                history = self.market_data.iloc[:i + 1]
                signals = agent.generate_signals(history, prices)

                for asset, sig in signals.items():
                    if asset not in assets:
                        continue
                    if isinstance(sig, tuple):
                        action, desired_value = sig[0], float(sig[1])
                    else:
                        action, desired_value = sig, 0.0

                    price = prices.get(asset, None)
                    if price is None or pd.isna(price) or price <= 0:
                        continue
                    price = float(price)

                    if action == 'BUY':
                        if desired_value <= 0:
                            pv = self.broker.get_portfolio_value(agent.name, prices)
                            desired_value = self.config['max_position_size'] * pv
                        self.broker.execute_order(
                            agent.name, asset, 'BUY',
                            desired_value, price, date,
                        )
                    elif action == 'SELL':
                        acct = self.broker.get_account(agent.name)
                        shares = acct.positions.get(asset, 0)
                        if shares > 0:
                            self.broker.execute_order(
                                agent.name, asset, 'SELL',
                                shares * price, price, date,
                            )

            # ---- 2. mark-to-market ----
            self.broker.mark_to_market(prices, date)
            daily_pool_values.append(self.broker.get_total_value(prices))

            # ---- 3. rebalance? ----
            if self._should_rebalance(date, last_rebalance):
                metrics = self.broker.get_all_metrics()
                pool = self.broker.get_total_value(prices)
                new_alloc = compute_new_allocations(metrics, pool, self.config)

                # Adapt params BEFORE redistribution so agents see
                # metrics from the period that just ended.
                for agent in self.agents:
                    before = agent.get_params()
                    agent.adapt_parameters(metrics[agent.name])
                    after = agent.get_params()
                    for k in before:
                        if before[k] != after[k]:
                            param_changes.append({
                                'date': str(date.date()),
                                'agent': agent.name,
                                'param': k,
                                'old': round(float(before[k]), 6),
                                'new': round(float(after[k]), 6),
                            })

                self.broker.redistribute_capital(new_alloc, prices, date)
                last_rebalance = date

        # ---- final metrics ----
        final_pool = daily_pool_values[-1] if daily_pool_values else 0.0
        initial = self.config['initial_capital']

        return {
            'portfolio_values': daily_pool_values,
            'dates': [str(d.date()) for d in trading_days],
            'agent_metrics': self.broker.get_all_metrics(),
            'allocation_history': self.broker.capital_history,
            'trade_log': [asdict(o) for o in self.broker.trade_log],
            'param_changes': param_changes,
            'final_return': (final_pool - initial) / initial if initial > 0 else 0.0,
            'benchmark_return': calculate_benchmark_return(
                self.market_data, self.config['benchmark']
            ),
            'final_pool': final_pool,
            'config': {
                k: v for k, v in self.config.items()
                if isinstance(v, (str, int, float, bool, list, type(None)))
            },
        }