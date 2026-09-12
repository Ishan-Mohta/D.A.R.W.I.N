"""
Strategy registry + agent factory.

Allows the user to pick strategies by name or count. If more agents are
requested than there are strategies, the registry cycles strategies and
suffixes their names with _1, _2, ... and jitters their initial params
so they diverge over time.

RISK APPETITE:
  Each agent samples its OWN risk level from a Gaussian distribution
  centered on the user's risk appetite, with std-dev controlled by
  'risk_distribution_std'. This produces a bell curve of agent risk
  levels around the user's choice.
"""

import random
from typing import List, Union

from src.agents.base_agent import BaseAgent
from src.agents.contrarian_agent import ContrarianAgent
from src.agents.mean_reversion_agent import MeanReversionAgent
from src.agents.ml_agent import MLAgent
from src.agents.momentum_agent import MomentumAgent
from src.agents.risk_parity_agent import RiskParityAgent
from src.utils.logger import get_logger

logger = get_logger(__name__)


STRATEGY_REGISTRY = {
    'momentum': MomentumAgent,
    'mean_reversion': MeanReversionAgent,
    'ml': MLAgent,
    'risk_parity': RiskParityAgent,
    'contrarian': ContrarianAgent,
}


def _suffix(name: str, counts: dict) -> str:
    counts[name] = counts.get(name, 0) + 1
    if counts[name] == 1:
        return name
    return f"{name}_{counts[name]}"


def build_agents(
    selection: Union[str, List[str], int],
    assets: List[str],
    user_risk_appetite: float = 5.0,
    risk_std: float = 0.7,
    risk_min: float = 0.5,
    risk_max: float = 10.0,
    seed: int = None,
) -> List[BaseAgent]:
    """
    Build a list of agents.

    selection: 'all' | list[str] | int
    user_risk_appetite: 0-10 scale, center of the risk distribution
    risk_std: std-dev of the Gaussian risk distribution
    risk_min / risk_max: clamp bounds for sampled risk
    seed: optional random seed for reproducibility

    Returns a list of instantiated agents. Each agent carries its own
    sampled risk_level.
    """
    if seed is not None:
        random.seed(seed)

    # ---- Determine which strategy names to instantiate ----
    names: List[str] = []
    if selection == 'all' or selection is None:
        names = list(STRATEGY_REGISTRY.keys())
    elif isinstance(selection, list):
        names = list(selection)
    elif isinstance(selection, int):
        keys = list(STRATEGY_REGISTRY.keys())
        # Cycle through strategies to reach the requested count
        names = [keys[i % len(keys)] for i in range(selection)]
        logger.info(f"Cycling {len(keys)} strategies to build {selection} agents")
    else:
        raise ValueError(f"Unsupported agent_selection: {selection!r}")

    # ---- Pad to at least 3 agents ----
    if len(names) < 3:
        logger.warning(f"Requested {len(names)} agents; padding to 3")
        keys = list(STRATEGY_REGISTRY.keys())
        while len(names) < 3:
            names.append(keys[len(names) % len(keys)])

    # ---- Instantiate ----
    counts: dict = {}
    agents: List[BaseAgent] = []
    for raw in names:
        if raw not in STRATEGY_REGISTRY:
            logger.warning(f"Unknown strategy {raw!r}; skipping")
            continue
        cls = STRATEGY_REGISTRY[raw]
        agent_name = _suffix(raw, counts)
        jitter = counts[raw] > 1

        # ---- Sample this agent's risk level from the bell curve ----
        sampled_risk = random.gauss(user_risk_appetite, risk_std)
        sampled_risk = max(risk_min, min(risk_max, sampled_risk))

        agents.append(
            cls(agent_name, assets, jitter=jitter, risk_level=sampled_risk)
        )
        logger.info(
            f"Built {agent_name:20s} risk_level={sampled_risk:.2f} "
            f"(user set {user_risk_appetite:.1f}, std {risk_std:.2f})"
        )

    return agents