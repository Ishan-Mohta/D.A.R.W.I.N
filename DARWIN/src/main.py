"""
Entry point for the Evolutionary Investment Platform.

CLI usage:
    python -m src.main
    python -m src.main --num-agents 7 --capital 250000 --freq M
    python -m src.main --risk 7.5 --risk-std 1.2
    python -m src.main --agents momentum,ml,contrarian

The RISK APPETITE and RISK DEVIATION are the two flagship user inputs.
They control the center and spread of the Gaussian distribution from
which each agent samples its own risk level.
"""

import argparse
import json
import os
from typing import Dict

from src.adaptation.reallocation import compute_new_allocations   # noqa: F401
from src.agents.registry import build_agents
from src.engine.backtester import Backtester
from src.engine.broker import CentralBroker
from src.utils import config as cfg
from src.utils.data_fetcher import fetch_market_data
from src.utils.logger import get_logger

logger = get_logger(__name__)


# =====================================================================
# JSON SAFETY — convert numpy/pandas types to native Python
# =====================================================================
def _to_native(obj):
    """Recursively convert numpy/pandas types to native Python for JSON."""
    import numpy as np
    import pandas as pd

    if isinstance(obj, dict):
        return {str(k): _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_native(v) for v in obj]
    if isinstance(obj, (pd.Timestamp, pd.Timedelta)):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        return str(obj)


# =====================================================================
# MAIN ENTRY POINT
# =====================================================================
def run_platform(config_overrides: Dict = None) -> Dict:
    """
    Run a complete backtest.

    Key override keys:
        initial_capital         - float, starting pool
        num_agents              - int, how many bots (3-15)
        agent_selection         - 'all' | list[str] | int
        user_risk_appetite      - float, 0-10, CENTER of risk bell curve
        risk_distribution_std   - float, WIDTH of risk bell curve
        rebalance_freq          - 'W' | 'M' | 'Q'
        assets                  - list[str], tickers
    """
    cfg_dict = cfg.merge_config(config_overrides)

    logger.info("=" * 72)
    logger.info("EVOLUTIONARY INVESTMENT PLATFORM")
    logger.info("=" * 72)
    logger.info(f"Starting capital:       ${cfg_dict['initial_capital']:,.2f}")
    logger.info(f"Number of agents:       {cfg_dict['num_agents']}")
    logger.info(f"Risk appetite (center): {cfg_dict['user_risk_appetite']:.2f} / 10")
    logger.info(f"Risk deviation (std):   {cfg_dict['risk_distribution_std']:.2f}")
    logger.info(f"Rebalance frequency:    {cfg_dict['rebalance_freq']}")
    logger.info(f"Assets:                 {cfg_dict['assets']}")
    logger.info("-" * 72)

    # ---- 1. Data ----
    market_data = fetch_market_data(
        assets=cfg_dict['assets'],
        period=cfg_dict['data_period'],
        cache_path=cfg_dict['cache_path'],
    )
    logger.info(f"Loaded {len(market_data)} trading days")

    # ---- 2. Agents (with sampled risk levels) ----
    # If the user did not pick specific strategies, use num_agents from the slider
    selection = cfg_dict.get('agent_selection')
    if selection in (None, 'all'):
        selection = cfg_dict['num_agents']
    
    agents = build_agents(
        selection=selection,
        assets=cfg_dict['assets'],
        user_risk_appetite=cfg_dict['user_risk_appetite'],
        risk_std=cfg_dict['risk_distribution_std'],
        risk_min=cfg_dict['risk_clamp_min'],
        risk_max=cfg_dict['risk_clamp_max'],
    )
    logger.info(f"Built {len(agents)} agents")
    for a in agents:
        logger.info(
            f"  {a.name:20s} risk={a.risk_level:.2f} "
            f"max_pos={a.max_position_size:.2%} "
            f"cash_buf={a.min_cash_buffer:.2%} "
            f"aggression={a.signal_aggression:.2f}"
        )

    # ---- 3. Per-agent risk parameters for the broker ----
    risk_params = {
        a.name: {
            'risk_level': a.risk_level,
            'max_position_size': a.max_position_size,
            'min_cash_buffer': a.min_cash_buffer,
        }
        for a in agents
    }

    # ---- 4. Broker ----
    broker = CentralBroker(
        agent_ids=[a.name for a in agents],
        initial_capital=cfg_dict['initial_capital'],
        assets=cfg_dict['assets'],
        config=cfg_dict,
        risk_params=risk_params,
    )

    # ---- 5. Run ----
    bt = Backtester(agents, market_data, broker, cfg_dict)
    results = bt.run()

    # ---- 6. Attach per-agent risk levels to results for the dashboard ----
    for aid, m in results['agent_metrics'].items():
        m['risk_level'] = risk_params[aid]['risk_level']
        m['max_position_size'] = risk_params[aid]['max_position_size']
        m['min_cash_buffer'] = risk_params[aid]['min_cash_buffer']

    results['risk_appetite'] = cfg_dict['user_risk_appetite']
    results['risk_std'] = cfg_dict['risk_distribution_std']

    # ---- 7. Save ----
    os.makedirs(os.path.dirname(cfg_dict['results_path']) or '.', exist_ok=True)
    with open(cfg_dict['results_path'], 'w') as f:
        json.dump(_to_native(results), f, indent=2)
    logger.info(f"Results saved to {cfg_dict['results_path']}")

    # ---- 8. Print summary ----
    print("\n" + "=" * 88)
    print("EVOLUTIONARY INVESTMENT PLATFORM — RESULTS")
    print("=" * 88)
    print(f"Starting capital:       ${cfg_dict['initial_capital']:,.2f}")
    print(f"Final pool:             ${results['final_pool']:,.2f}")
    print(f"Total return:           {results['final_return'] * 100:+.2f}%")
    print(f"Benchmark ({cfg_dict['benchmark']}):      "
          f"{results['benchmark_return'] * 100:+.2f}%")
    print(f"Risk appetite (center): {cfg_dict['user_risk_appetite']:.2f} / 10")
    print(f"Risk deviation (std):   {cfg_dict['risk_distribution_std']:.2f}")
    print("-" * 88)
    print(f"{'Agent':<18}{'Risk':>7}{'Return':>11}{'Sharpe':>10}"
          f"{'MaxDD':>10}{'WinRate':>10}{'FinalCap':>14}{'Trades':>8}")
    print("-" * 88)
    for aid, m in sorted(results['agent_metrics'].items(),
                         key=lambda x: x[1]['return'], reverse=True):
        final_cap = broker.accounts[aid].portfolio_history[-1] \
            if broker.accounts[aid].portfolio_history else 0.0
        print(f"{aid:<18}"
              f"{m['risk_level']:>7.2f}"
              f"{m['return'] * 100:>10.2f}%"
              f"{m['sharpe']:>10.2f}"
              f"{m['max_drawdown'] * 100:>9.2f}%"
              f"{m['win_rate'] * 100:>9.1f}%"
              f"${final_cap:>13,.2f}"
              f"{m['num_trades']:>8d}")
    print("=" * 88 + "\n")

    return results


# =====================================================================
# CLI
# =====================================================================
def _cli():
    parser = argparse.ArgumentParser(
        description="Evolutionary Investment Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.main
  python -m src.main --num-agents 7 --capital 250000
  python -m src.main --risk 7.5 --risk-std 1.2
  python -m src.main --risk 2 --risk-std 0.5
  python -m src.main --agents momentum,ml,contrarian
        """,
    )
    parser.add_argument('--num-agents', type=int, default=None,
                        help="Number of agents (3-15)")
    parser.add_argument('--capital', type=float, default=None,
                        help="Starting capital")
    parser.add_argument('--freq', type=str, default=None,
                        choices=['W', 'M', 'Q'],
                        help="Rebalance frequency")
    parser.add_argument('--risk', type=float, default=None,
                        help="Risk appetite, 0-10 (CENTER of bell curve)")
    parser.add_argument('--risk-std', type=float, default=None,
                        help="Risk deviation (WIDTH of bell curve)")
    parser.add_argument('--agents', type=str, default=None,
                        help="Comma-separated strategy names")
    parser.add_argument('--assets', type=str, default=None,
                        help="Comma-separated tickers")
    parser.add_argument('--refresh-data', action='store_true',
                        help="Ignore cache and re-download")
    args = parser.parse_args()

    overrides = {}

    if args.num_agents is not None:
        overrides['num_agents'] = args.num_agents
        overrides['agent_selection'] = args.num_agents

    if args.capital is not None:
        overrides['initial_capital'] = args.capital

    if args.freq is not None:
        overrides['rebalance_freq'] = args.freq

    if args.risk is not None:
        if not 0 <= args.risk <= 10:
            parser.error("--risk must be between 0 and 10")
        overrides['user_risk_appetite'] = args.risk

    if args.risk_std is not None:
        if args.risk_std < 0:
            parser.error("--risk-std must be >= 0")
        overrides['risk_distribution_std'] = args.risk_std

    if args.agents:
        overrides['agent_selection'] = [
            a.strip() for a in args.agents.split(',') if a.strip()
        ]

    if args.assets:
        overrides['assets'] = [a.strip().upper() for a in args.assets.split(',')]
        overrides['benchmark'] = overrides['assets'][-1]

    if args.refresh_data:
        from src.utils import config as c
        p = c.DEFAULTS['cache_path']
        if os.path.exists(p):
            os.remove(p)

    run_platform(overrides)


if __name__ == '__main__':
    _cli()