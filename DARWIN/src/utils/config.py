"""
Global configuration for the Evolutionary Investment Platform.

Everything here is a DEFAULT. The user overrides at runtime by passing
a dict to run_platform(). No module may hardcode a literal like 5 or
100_000 — all such values must be read from here.

RISK APPETITE:
  The user picks a single value (0-10) representing overall risk.
  Each agent samples its OWN risk level from a Gaussian distribution
  centered on the user's value, with std-dev controlled by
  'risk_distribution_std'. Higher std = more diversity in the fleet.
"""

DEFAULTS = {
    # ---- Capital (user-overridable) ----
    'initial_capital': 100_000,
    'num_agents': 5,
    'agent_selection': 'all',

    # ---- Universe ----
    'assets': ['AAPL', 'GOOGL', 'MSFT', 'SPY'],
    'benchmark': 'SPY',

    # ---- Time ----
    'data_period': '1y',
    'rebalance_freq': 'M',
    'execution_freq': 'D',
    'trading_days_per_year': 252,

    # ---- Risk appetite (USER-FACING) ----
    # 0-10 scale. Higher = more aggressive.
    # Individual agents sample from N(user_risk_appetite, risk_std^2).
    'user_risk_appetite': 5.0,
    'risk_distribution_std': 0.7,     # spread of the bell curve
    'risk_clamp_min': 0.5,            # no agent may go below this
    'risk_clamp_max': 10.0,           # nor above this

    # ---- Guardrails (global fallbacks; per-agent overrides apply) ----
    'max_agent_drawdown': -0.20,
    'max_position_size': 0.15,        # fallback if agent has no risk
    'min_cash_buffer': 0.02,          # fallback if agent has no risk
    'min_agent_capital_pct': 0.01,

    # ---- Evolution ----
    'allocation_sensitivity': 1.0,
    'ml_history_window': 60,

    # ---- Paths ----
    'cache_path': 'data/historical_data.pkl',
    'results_path': 'data/results.json',
    'log_level': 'INFO',
}


def merge_config(overrides: dict = None) -> dict:
    """Return a fresh config dict with user overrides applied."""
    cfg = dict(DEFAULTS)
    if overrides:
        cfg.update(overrides)
    return cfg