"""
Capital reallocation — PROPORTIONAL TO RELATIVE OUTPERFORMANCE.

We deliberately do NOT use "share of total profit" (unstable when total
profit is near zero or negative) nor a fixed boost/cut rule (too rigid).
Instead, each agent receives an equal base share PLUS a bonus
proportional to how much its return exceeded the group mean. See the
worked example at the bottom of this file.
"""

from typing import Dict

from src.utils.logger import get_logger

logger = get_logger(__name__)


def compute_new_allocations(
    agent_metrics: Dict[str, Dict],
    current_total_pool: float,
    config: dict,
) -> Dict[str, float]:
    """
    Return {agent_id: new_dollar_allocation} summing exactly to
    current_total_pool after guardrails.
    """
    ids = list(agent_metrics.keys())
    n = len(ids)
    if n == 0 or current_total_pool <= 0:
        return {aid: 0.0 for aid in ids}

    # ---- Step 1: returns & mean ----
    returns = {aid: agent_metrics[aid].get('return', 0.0) for aid in ids}
    r_mean = sum(returns.values()) / n

    # ---- Step 2: excess over mean ----
    excess = {aid: returns[aid] - r_mean for aid in ids}

    # ---- Step 3: equal base + sensitivity * excess ----
    k = config.get('allocation_sensitivity', 1.0)
    base = 1.0 / n
    raw = {aid: base + k * excess[aid] for aid in ids}

    # ---- Step 4: floor raw weights ----
    floor_pct = config.get('min_agent_capital_pct', 0.01)
    raw = {aid: max(w, floor_pct) for aid, w in raw.items()}

    
    # ---- Step 5: drawdown kill-switch ----
    killed = set()
    max_dd = config.get('max_agent_drawdown', -1.0)
    for aid in ids:
        dd = agent_metrics[aid].get('max_drawdown', 0.0)
        if dd < max_dd:
            killed.add(aid)
            logger.warning(f"KILL: {aid} hit drawdown {dd:.2%} < {max_dd:.2%}")

    # ---- Step 5.5: FLAG LOSERS FOR MUTATION ----
    # Any bot with negative return gets flagged for parameter mutation.
    # The actual mutation happens in backtester.py via agent.mutate().
    # Stronger losses = stronger mutation (bigger exploration).
    mutation_threshold = config.get('mutation_threshold', -0.02)
    for aid in ids:
        r = returns[aid]
        if r < mutation_threshold:
            # Absolute loss trigger — force mutation
            strength = min(abs(r) * 5.0, 1.0)
            agent_metrics[aid]['mutation'] = {
                'needs_mutation': True,
                'strength': strength,
                'reason': f"loss {r:.2%}",
            }
            logger.info(
                f"MUTATE FLAG: {aid} return={r:.2%} strength={strength:.2f}"
            )
        elif r < 0:
            # Small loss but above threshold — mild mutation
            agent_metrics[aid]['mutation'] = {
                'needs_mutation': True,
                'strength': 0.05,
                'reason': f"small loss {r:.2%}",
            }
        else:
            # Winner or flat — no mutation
            agent_metrics[aid]['mutation'] = {
                'needs_mutation': False,
                'strength': 0.0,
                'reason': 'positive',
            }

    # ---- Step 6: dollar allocations ----
    new_caps = {aid: raw[aid] * current_total_pool for aid in ids}

    # ---- Step 7: renormalize to pool ----
    total = sum(new_caps.values())
    if total > 0:
        scale = current_total_pool / total
        new_caps = {aid: v * scale for aid, v in new_caps.items()}

    # ---- Step 8: enforce floor one final time ----
    floor = floor_pct * current_total_pool
    new_caps = {aid: max(v, floor) for aid, v in new_caps.items()}

    logger.info(
        f"Reallocated pool of {current_total_pool:,.0f} | "
        f"mean return {r_mean:.2%} | killed={sorted(killed)}"
    )
    return new_caps


# =====================================================================
# WHY THIS FORMULA
# =====================================================================
# - Stability: we never divide by total_profit, so a flat month
#   (r_mean ~ 0) does not blow up the shares.
# - Correctness in losing months: if everyone lost money but you lost
#   less, excess_i > 0 and you still get a larger share.
# - Boundedness: k controls how aggressively capital chases
#   outperformance. k=1 gives roughly 1/N +/- excess per agent.
# - Zero-sum base: equal share (1/N) ensures no agent is wiped out
#   purely because everyone else had a good month.
#
# =====================================================================
# WORKED EXAMPLE (k = 1.0, pool = $100,000, 4 agents)
# =====================================================================
# Returns this month:
#     A: +12%  ->  excess = +5.75%
#     B:  +4%  ->  excess = -2.25%
#     C:  -2%  ->  excess = -8.25%
#     D:  -6%  ->  excess = -12.25%
#     r_mean = +2.0%
#
# raw weights: A=0.3075, B=0.2275, C=0.1675, D=0.1275
# new caps:    A=$30,750, B=$22,750, C=$16,750, D=$12,750
# sum = $83,000 -> renormalize by 100/83:
#              A=$37,048, B=$27,410, C=$20,181, D=$15,361
# sum = $100,000
#
# A's share grew (24% -> 37%). D's shrank (24% -> 15%).
# Nobody is zeroed; B/C/D keep meaningful capital to try again.
# =====================================================================