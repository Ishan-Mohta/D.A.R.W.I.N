"""Reusable dashboard components (optional, imported by app.py if needed)."""

import pandas as pd
import plotly.express as px


def metric_row(st, results):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Final Pool", f"${results['final_pool']:,.0f}")
    c2.metric("Total Return", f"{results['final_return'] * 100:+.2f}%")
    c3.metric("Starting Capital",
              f"${results['config']['initial_capital']:,.0f}")
    c4.metric(f"Benchmark", f"{results['benchmark_return'] * 100:+.2f}%")


def leaderboard_df(metrics, final_caps):
    rows = []
    for aid, m in metrics.items():
        rows.append({
            'Agent': aid,
            'Return %': round(m['return'] * 100, 2),
            'Sharpe': round(m['sharpe'], 2),
            'Max DD %': round(m['max_drawdown'] * 100, 2),
            'Win Rate %': round(m['win_rate'] * 100, 1),
            'Final Cap': round(final_caps.get(aid, 0.0), 2),
            'Trades': m['num_trades'],
        })
    df = pd.DataFrame(rows).sort_values('Return %', ascending=False).reset_index(drop=True)
    df.insert(0, 'Rank', range(1, len(df) + 1))
    return df


def returns_bar(df):
    return px.bar(df, x='Agent', y='Return %', color='Agent', text='Return %')