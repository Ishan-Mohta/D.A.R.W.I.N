"""
Streamlit dashboard for the Evolutionary Investment Platform.
Run: streamlit run dashboard/app.py
"""

import json
import os
import sys

import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

# Make src/ importable when running from project root
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')
))

from src.agents.registry import STRATEGY_REGISTRY
from src.main import run_platform

# =====================================================================
# PAGE CONFIG
# =====================================================================
st.set_page_config(
    page_title="D.A.R.W.I.N.",
    layout="wide",
)

# =====================================================================
# CHECK QUERY PARAM — user clicked the splash button
# Streamlit < 1.30 API (returns lists)
# =====================================================================
qp = st.experimental_get_query_params()
if qp.get("enter") == ["1"]:
    st.session_state.splash_done = True

# =====================================================================
# GLOBAL SIDEBAR HOVER EFFECTS
# =====================================================================
st.markdown("""
<style>
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4 {
        transition: all 0.25s ease-in-out;
        cursor: pointer;
        display: inline-block;
        transform-origin: left center;
    }
    section[data-testid="stSidebar"] h1:hover,
    section[data-testid="stSidebar"] h2:hover,
    section[data-testid="stSidebar"] h3:hover,
    section[data-testid="stSidebar"] h4:hover {
        transform: scale(1.12);
        color: #a7ef9e !important;
        text-shadow: 0 0 12px rgba(167, 239, 158, 0.7),
                     0 0 24px rgba(167, 239, 158, 0.4);
        letter-spacing: 0.5px;
    }
    section[data-testid="stSidebar"] p strong {
        transition: all 0.25s ease-in-out;
        cursor: pointer;
        display: inline-block;
        transform-origin: left center;
    }
    section[data-testid="stSidebar"] p strong:hover {
        transform: scale(1.10);
        color: #a7ef9e !important;
        text-shadow: 0 0 10px rgba(167, 239, 158, 0.6);
    }
    section[data-testid="stSidebar"] .stSlider:hover {
        transform: scale(1.02);
        transition: transform 0.2s ease;
    }
</style>
""", unsafe_allow_html=True)

# =====================================================================
# SPLASH SCREEN
# =====================================================================
if "splash_done" not in st.session_state:
    st.session_state.splash_done = False

if not st.session_state.splash_done:
    # Splash visual with an inline HTML <a> link that sets ?enter=1
    st.markdown("""
    <style>
      section[data-testid="stSidebar"] { display: none !important; }
      header[data-testid="stHeader"] { display: none !important; }
      footer { display: none !important; }
      .block-container {
          padding: 0 !important;
          max-width: 100% !important;
          margin: 0 !important;
      }

      /* Full-screen splash */
      .splash-wrap {
        position: fixed;
        inset: 0;
        background: radial-gradient(circle at center, #0a2a0f 0%, #030d05 70%);
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        z-index: 100;
        animation: fadeIn 0.8s ease-in;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      }
      @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }

      .splash-title {
        font-size: clamp(2.5rem, 8vw, 6rem);
        font-weight: 900;
        letter-spacing: 0.15em;
        color: #a7ef9e;
        text-shadow: 0 0 20px rgba(167,239,158,0.8),
                     0 0 45px rgba(167,239,158,0.4);
        margin-bottom: 0.25em;
        animation: pulse 3s ease-in-out infinite;
      }
      @keyframes pulse {
        0%, 100% { text-shadow: 0 0 20px rgba(167,239,158,0.8),
                                0 0 45px rgba(167,239,158,0.4); }
        50%      { text-shadow: 0 0 30px rgba(167,239,158,1),
                                0 0 60px rgba(167,239,158,0.6); }
      }

      .splash-sub {
        font-size: 1rem;
        letter-spacing: 0.4em;
        color: #7ab87a;
        text-transform: uppercase;
        margin-bottom: 3em;
        opacity: 0.85;
      }

      .splash-bar {
        width: 300px;
        height: 3px;
        background: #0a2a0f;
        border-radius: 2px;
        overflow: hidden;
        position: relative;
        margin-bottom: 3em;
      }
      .splash-bar::after {
        content: "";
        position: absolute;
        left: 0; top: 0; bottom: 0;
        width: 40%;
        background: linear-gradient(90deg, transparent, #a7ef9e, transparent);
        animation: slide 1.6s ease-in-out infinite;
      }
      @keyframes slide {
        0%   { left: -40%; }
        100% { left: 100%; }
      }

      /* The entry link styled as a glowing button */
      .enter-link {
        display: inline-block;
        background: linear-gradient(135deg, #0f3d14, #0a2a0f);
        border: 2px solid #a7ef9e;
        color: #a7ef9e !important;
        font-weight: 700;
        letter-spacing: 0.15em;
        font-size: 1rem;
        padding: 0.9em 2.4em;
        border-radius: 8px;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 0 15px rgba(167,239,158,0.4),
                    0 0 30px rgba(167,239,158,0.2);
        text-transform: uppercase;
        text-decoration: none !important;
        font-family: inherit;
      }
      .enter-link:hover {
        box-shadow: 0 0 25px rgba(167,239,158,0.9),
                    0 0 50px rgba(167,239,158,0.5);
        transform: scale(1.06);
        border-color: #c8ffb8;
        color: #c8ffb8 !important;
        text-decoration: none !important;
      }
      .enter-link:active {
        transform: scale(0.98);
      }
    </style>

    <div class="splash-wrap">
      <div class="splash-title">D.A.R.W.I.N</div>
      <div class="splash-sub">Evolutionary Investment Platform</div>
      <div class="splash-bar"></div>
      <a class="enter-link" href="?enter=1" target="_self"> Proceed</a>
    </div>
    """, unsafe_allow_html=True)

    st.stop()

# Clear the query param now that we've entered, so refreshes stay clean
if st.session_state.splash_done and qp.get("enter") == ["1"]:
    st.experimental_set_query_params()

# =====================================================================
# HERO BANNER
# =====================================================================
_banner_path = os.path.join(os.path.dirname(__file__), "hero_banner.html")
with open(_banner_path, "r", encoding="utf-8") as _f:
    components.html(_f.read(), height=340)

st.caption("Multi-agent backtesting where bots compete for capital and evolve.")

# =====================================================================
# SIDEBAR CONFIG
# =====================================================================
st.sidebar.header("Configuration")

num_agents = st.sidebar.slider(
    "Number of agents", 3, 15, 5,
    help="How many bots compete (duplicates auto-suffixed)"
)

strategy_choice = st.sidebar.multiselect(
    "Strategies (empty = use agent count)",
    options=list(STRATEGY_REGISTRY.keys()),
    default=[],
)

capital = st.sidebar.number_input(
    "Starting capital ($)",
    min_value=1_000.0, value=100_000.0, step=10_000.0,
)

freq = st.sidebar.radio(
    "Rebalance frequency",
    options=['W', 'M', 'Q'],
    index=1,
    format_func=lambda x: {'W': 'Weekly', 'M': 'Monthly', 'Q': 'Quarterly'}[x],
)

st.sidebar.markdown("---")
st.sidebar.subheader("Risk Appetite")
st.sidebar.caption("These two values control the entire fleet's behavior.")

user_risk = st.sidebar.slider(
    "Risk appetite (0-10)",
    min_value=0.0, max_value=10.0, value=5.0, step=0.5,
    help="CENTER of the bell curve. Each bot samples its own risk "
         "level from a distribution centered here."
)

risk_std = st.sidebar.slider(
    "Risk deviation (std-dev)",
    min_value=0.1, max_value=2.0, value=0.7, step=0.1,
    help="WIDTH of the bell curve. Higher = more diversity in the "
         "fleet's risk levels."
)

st.sidebar.markdown("---")

assets_input = st.sidebar.text_input(
    "Assets (comma-separated)", value="AAPL,GOOGL,MSFT,SPY"
)

run_clicked = st.sidebar.button("Run Backtest", type="primary")

# =====================================================================
# STATE
# =====================================================================
if 'results' not in st.session_state:
    st.session_state.results = None

if run_clicked:
    assets = [a.strip().upper() for a in assets_input.split(',') if a.strip()]
    if len(assets) < 2:
        st.sidebar.error("Need at least 2 assets.")
    else:
        selection = strategy_choice if strategy_choice else num_agents
        overrides = {
            'num_agents': num_agents,
            'agent_selection': selection,
            'initial_capital': capital,
            'rebalance_freq': freq,
            'assets': assets,
            'benchmark': assets[-1],
            'user_risk_appetite': user_risk,
            'risk_distribution_std': risk_std,
        }
        with st.spinner("Running backtest..."):
            try:
                results = run_platform(overrides)
                st.session_state.results = results
            except Exception as e:
                st.error(f"Backtest failed: {e}")
                st.exception(e)

# =====================================================================
# RESULTS
# =====================================================================
results = st.session_state.results

if results is None:
    st.info("Configure your platform in the sidebar and click **Run Backtest**.")
    st.stop()

st.subheader("Risk Distribution of the Fleet")
st.caption(
    f"User set risk = **{results['risk_appetite']:.1f}** with deviation "
    f"**{results['risk_std']:.1f}**. Each bot sampled its own risk from "
    f"this distribution."
)

risk_data = pd.DataFrame([
    {'Agent': aid, 'Risk': m.get('risk_level', 5.0)}
    for aid, m in results['agent_metrics'].items()
]).sort_values('Risk', ascending=False).reset_index(drop=True)

fig_risk = px.bar(
    risk_data, x='Agent', y='Risk', color='Risk',
    color_continuous_scale='RdYlGn_r',
    text='Risk',
)
fig_risk.update_traces(texttemplate='%{text:.2f}', textposition='outside')
fig_risk.add_hline(
    y=results['risk_appetite'], line_dash="dash", line_color="blue",
    annotation_text=f"Your setting: {results['risk_appetite']:.1f}",
)
fig_risk.update_layout(yaxis_range=[0, 10.5])
st.plotly_chart(fig_risk, use_container_width=True)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Final Pool", f"${results['final_pool']:,.0f}")
c2.metric("Total Return", f"{results['final_return'] * 100:+.2f}%")
c3.metric("Starting Capital",
          f"${results['config']['initial_capital']:,.0f}")
c4.metric(f"Benchmark ({results['config']['benchmark']})",
          f"{results['benchmark_return'] * 100:+.2f}%")

st.subheader("Agent Leaderboard")

metrics = results['agent_metrics']
alloc_hist = results['allocation_history']

final_caps = {}
if alloc_hist:
    last = alloc_hist[-1]['agents']
    for aid, d in last.items():
        final_caps[aid] = d['new_cap']
else:
    n = len(metrics)
    for aid in metrics:
        final_caps[aid] = results['config']['initial_capital'] / n

rows = []
for aid, m in metrics.items():
    rows.append({
        'Agent': aid,
        'Risk': round(m.get('risk_level', 5.0), 2),
        'MaxPos %': round(m.get('max_position_size', 0.15) * 100, 1),
        'Return %': round(m['return'] * 100, 2),
        'Sharpe': round(m['sharpe'], 2),
        'Max DD %': round(m['max_drawdown'] * 100, 2),
        'Win Rate %': round(m['win_rate'] * 100, 1),
        'Final Cap': round(final_caps.get(aid, 0.0), 2),
        'Trades': m['num_trades'],
    })
lb = (pd.DataFrame(rows)
        .sort_values('Return %', ascending=False)
        .reset_index(drop=True))
lb.insert(0, 'Rank', range(1, len(lb) + 1))
st.dataframe(lb, use_container_width=True)

col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Returns by Agent")
    fig = px.bar(lb, x='Agent', y='Return %', color='Agent', text='Return %')
    fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    st.plotly_chart(fig, use_container_width=True)

with col_b:
    st.subheader("Risk vs Return")
    fig = px.scatter(
        lb, x='Risk', y='Return %',
        size='Final Cap', color='Agent', hover_name='Agent',
        text='Agent',
    )
    fig.add_vline(x=results['risk_appetite'], line_dash="dash",
                  line_color="gray",
                  annotation_text="Your setting")
    st.plotly_chart(fig, use_container_width=True)

col_c, col_d = st.columns(2)

with col_c:
    st.subheader("Final Capital Distribution")
    fig = px.pie(lb, names='Agent', values='Final Cap', hole=0.4)
    st.plotly_chart(fig, use_container_width=True)

with col_d:
    st.subheader("Total Portfolio Value")
    df = pd.DataFrame({
        'Date': pd.to_datetime(results['dates']),
        'Value': results['portfolio_values'],
    })
    fig = px.line(df, x='Date', y='Value')
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Capital Allocation Over Time")
if alloc_hist:
    timeline_rows = []
    for snap in alloc_hist:
        for aid, d in snap['agents'].items():
            timeline_rows.append({
                'Date': pd.to_datetime(snap['timestamp']),
                'Agent': aid,
                'Capital': d['new_cap'],
            })
    tl = pd.DataFrame(timeline_rows)
    fig = px.line(tl, x='Date', y='Capital', color='Agent', markers=True)
    st.plotly_chart(fig, use_container_width=True)

st.subheader("🧬 Evolution Timeline")
if alloc_hist:
    evo_rows = []
    for snap in alloc_hist:
        for aid, d in snap['agents'].items():
            evo_rows.append({
                'Date': snap['timestamp'],
                'Agent': aid,
                'Old Cap': d['old_cap'],
                'New Cap': d['new_cap'],
                'Return %': round(d['return'] * 100, 2),
                'Profit': d['profit'],
            })
    st.dataframe(pd.DataFrame(evo_rows), use_container_width=True)

st.subheader("Trade Log (last 100)")
trades = results['trade_log']
if trades:
    tdf = pd.DataFrame(trades[-100:])
    cols = ['timestamp', 'agent_id', 'asset', 'side',
            'filled_shares', 'price', 'filled_value', 'status', 'reason']
    cols = [c for c in cols if c in tdf.columns]
    st.dataframe(tdf[cols], use_container_width=True)
else:
    st.write("No trades executed.")

st.subheader("Parameter Drift")
pc = results.get('param_changes', [])
if pc:
    st.dataframe(pd.DataFrame(pc), use_container_width=True)
else:
    st.write("No parameter changes recorded.")

st.download_button(
    "⬇️ Download results.json",
    data=json.dumps(results, indent=2, default=str),
    file_name="results.json",
    mime="application/json",
)