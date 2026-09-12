"""Dashboard-facing query wrapper over the broker's state."""

from typing import Dict, List

import pandas as pd


class PortfolioView:
    """Read-only view of the broker for dashboards."""

    def __init__(self, broker):
        self.broker = broker

    def leaderboard(self) -> pd.DataFrame:
        metrics = self.broker.get_all_metrics()
        rows = []
        for aid, m in metrics.items():
            rows.append({
                'Agent': aid,
                'Return %': round(m['return'] * 100, 2),
                'Sharpe': round(m['sharpe'], 2),
                'Max DD %': round(m['max_drawdown'] * 100, 2),
                'Win Rate %': round(m['win_rate'] * 100, 1),
                'Final Cap': round(
                    self.broker.accounts[aid].portfolio_history[-1]
                    if self.broker.accounts[aid].portfolio_history else 0.0, 2
                ),
                'Trades': m['num_trades'],
            })
        df = pd.DataFrame(rows).sort_values('Return %', ascending=False).reset_index(drop=True)
        df.insert(0, 'Rank', range(1, len(df) + 1))
        return df

    def allocation_timeline(self) -> List[Dict]:
        return self.broker.capital_history