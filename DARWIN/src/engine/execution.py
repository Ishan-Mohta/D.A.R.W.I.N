"""Order validation + fill logic used by CentralBroker."""

from typing import Dict, Tuple


def validate_and_cap_buy(
    desired_value: float,
    agent_portfolio_value: float,
    existing_position_value: float,
    agent_cash: float,
    max_position_size: float,
    min_cash_buffer: float,
) -> Tuple[float, str]:
    """
    Return (capped_dollar_amount, reason).
    Reasons: 'ok' | 'insufficient_cash' | 'max_position_exceeded'
             | 'both_capped'
    """
    max_pos_value = max_position_size * agent_portfolio_value
    room_position = max_pos_value - existing_position_value
    room_cash = agent_cash - min_cash_buffer * agent_portfolio_value

    if room_cash <= 0:
        return 0.0, 'insufficient_cash'
    if room_position <= 0:
        return 0.0, 'max_position_exceeded'

    capped = min(desired_value, room_position, room_cash)
    if capped <= 0:
        return 0.0, 'insufficient_cash'

    reason = 'ok'
    if capped < desired_value:
        if room_position < room_cash and room_position < desired_value:
            reason = 'max_position_exceeded'
        else:
            reason = 'insufficient_cash'
    return capped, reason