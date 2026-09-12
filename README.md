# D.A.R.W.I.N.

## Distributed Agentic Resource & Weighted Investment Network

### Natural Selection & Dynamic Capital Reallocation for Autonomous Trading Agents

Static trading algorithms break when market regimes shift.

**D.A.R.W.I.N.** treats portfolio management as a living ecosystem: autonomous AI agents compete for real-time capital. Winners scale, while underperformers mutate or starve.

---

# Demo Preview

![D.A.R.W.I.N. Dashboard](docs/demo.png)

## Live Evolution Command Center

| Capital Race | Top Performing Agents |
|---|---|
| Agent 1 — Momentum: 15% (↓ 10%) | 1. Volatility Hunter — Sharpe: 2.45 |
| Agent 2 — Mean Reversion: 25% (↑ 5%) | 2. Risk Parity — Sharpe: 1.95 |
| Agent 3 — Volatility Breakout: 40% (↑ 15%) | |
| Agent 4 — Risk Parity: 20% | Mutation Log |
| | Agent 1: Win rate < 40% |
| | Mutating RSI lookback: 14 → 5 |

---

# The Core Problem & Our Breakthrough

Traditional trading systems often rely on fixed strategies and allocations. When market conditions change, a strategy that previously performed well can become ineffective.

**D.A.R.W.I.N.** creates a closed-loop ecosystem where multiple autonomous agents continuously compete, adapt, and receive capital based on their performance.

| Traditional Trading Bots | D.A.R.W.I.N. |
|---|---|
| **Static Allocations:** Fixed asset weights regardless of market regime shifts | **Dynamic Capital Reallocation:** Capital is continuously re-weighted using a temperature-scaled Softmax allocation engine |
| **Manual Bot Tweaking:** Humans must adjust parameters after drawdowns | **Closed-Loop Self-Adaptation:** Agents analyze their performance and automatically mutate strategy parameters |
| **Single-Point Failure:** One model's blind spot can damage the entire portfolio | **Multi-Agent Ecosystem:** Multiple specialized agents compete while weak strategies are gradually starved |
| **Fixed Strategy Parameters:** Parameters remain unchanged | **Metacognitive Mutation:** Underperforming agents modify their own hyperparameters |
| **Reactive Risk Management:** Risk controls are often external to the strategy | **Deterministic Risk Firewall:** Independent guardrails verify decisions before execution |

---

# Track 1 Alignment

| Subtrack | Feature Implemented | Technical Mechanism |
|---|---|---|
| **Subtrack 1 — Agentic Workflows** | Autonomous Trading Swarm | Each agent follows an asynchronous `Observe → Plan → Act → Audit` execution pipeline |
| **Subtrack 2 — Self-Learning & Adaptation** | Dynamic Capital Reallocation & Mutation | Temperature-scaled Softmax reallocates capital according to agent performance; underperformers trigger hyperparameter mutation |
| **Subtrack 3 — Guardrails & Trust** | Risk Firewall & Circuit Breakers | LLM-generated prices are verified against market data within a defined tolerance; trade-level risk limits and portfolio drawdown circuit breakers prevent unsafe actions |

---

# System Architecture

```mermaid
flowchart TD

    A[Market Data] --> B[Simulation Engine]

    B --> C1[Momentum Agent]
    B --> C2[Mean Reversion Agent]
    B --> C3[Volatility Hunter]
    B --> C4[Risk Parity Agent]

    C1 --> D[Observe → Plan → Act → Audit]
    C2 --> D
    C3 --> D
    C4 --> D

    D --> E[Performance Evaluation]

    E --> F[Capital Reallocator]
    F --> G[Temperature-Scaled Softmax]

    G --> C1
    G --> C2
    G --> C3
    G --> C4

    E --> H{Underperforming?}

    H -->|Yes| I[Metacognitive Mutation]

    I --> C1
    I --> C2
    I --> C3
    I --> C4

    D --> J[Risk Firewall]

    J --> K{Risk Check}

    K -->|Approved| L[Execute Trade]
    K -->|Rejected| M[Circuit Breaker]

    L --> E
