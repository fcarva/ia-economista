"""
Market Regime Dashboard Page (Track 3)
======================================

Visualizes HMM-detected market regimes with:
- Regime timeline (color-coded)
- Transition probability matrix
- Current regime indicator with confidence

Part of the Bloomberg Terminal UX enhancement.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from cointegration_gnn.data.data_loader import DataLoader
from cointegration_gnn.features.regime_detection import (
    HMMRegimeDetector, 
    MarketRegime, 
    get_regime_style
)
from cointegration_gnn.config import default_config
from dashboard.utils import load_css, make_flexoki_chart

# Page Config
st.set_page_config(page_title="Market Regimes", page_icon="📊", layout="wide")
load_css()

st.title("📊 Market Regime Detection")
st.markdown("### Hidden Markov Model for Bull/Bear/Sideways Classification")

# Sidebar
with st.sidebar:
    st.header("⚙️ HMM Parameters")
    n_regimes = st.slider("Number of Regimes", 2, 4, 3)
    lookback = st.slider("Lookback Window (days)", 10, 60, 20)
    
    st.divider()
    st.caption("Uses Gaussian HMM with [return, volatility] emissions.")


# --- DATA LOADING ---
from cointegration_gnn.data.brazil_macro import BrazilMacroFetcher

# --- DATA LOADING ---
@st.cache_data
def load_data():
    """Load returns data for regime detection."""
    loader = DataLoader(tickers=default_config.data.tickers)
    prices, returns = loader.fetch_data(
        start=default_config.data.train_start,
        end=default_config.data.test_end
    )
    return prices, returns

@st.cache_data
def load_macro():
    """Load Brazil Macro indicators."""
    fetcher = BrazilMacroFetcher(cache_path="data/brazil_macro.csv")
    # Fetch ample history for context
    df = fetcher.fetch_all(start_date="2010-01-01") 
    return df

@st.cache_data
def detect_regimes(_returns, n_regimes, lookback):
    """Detect market regimes using HMM."""
    detector = HMMRegimeDetector(
        n_regimes=n_regimes,
        lookback=lookback,
    )
    result = detector.fit_predict(_returns)
    return result

# Load data
with st.spinner("Loading market & macro intelligence..."):
    prices, returns = load_data()
    macro_df = load_macro()

# Detect regimes
with st.spinner("Fitting HMM..."):
    result = detect_regimes(returns, n_regimes, lookback)

# --- MACRO LEDGER (The "RevNets" Style) ---
st.divider()
st.subheader("📑 The Macro Ledger")
st.caption("Fundamental economic indicators (BCB/IBGE) | Source of Truth")

# Prepare Ledger Data
latest_macro = macro_df.iloc[-1]
# Calculate MoM/YoY where applicable or use pre-calculated columns
# We have 'ipca_mom', 'pib_yoy', 'ibc_br_yoy'
# Format for display: Indicator | Value | Trend | Context

# Custom CSS for Ledger Table
st.markdown("""
<style>
    .ledger-table {
        font-family: 'JetBrains Mono', monospace;
        width: 100%;
        border-collapse: collapse;
        background-color: var(--sidebar-bg);
        border: 1px solid var(--border-color);
    }
    .ledger-table th {
        text-align: left;
        padding: 8px 12px;
        border-bottom: 2px solid var(--border-color);
        color: #6F6E69;
        font-size: 0.85rem;
        text-transform: uppercase;
    }
    .ledger-table td {
        padding: 8px 12px;
        border-bottom: 1px solid var(--border-color);
        color: var(--text-color);
    }
    .val-pos { color: #879A39; } /* Bullish/Positive */
    .val-neg { color: #D14D41; } /* Bearish/Negative */
    .val-neu { color: #6F6E69; } /* Neutral */
</style>
""", unsafe_allow_html=True)

# Helper to format value
def fmt_val(val, is_pct=True):
    if pd.isna(val): return "N/A"
    return f"{val:.2%}" if is_pct else f"{val:.4f}"

# Build rows
ledger_items = [
    ("SELIC (Meta)", latest_macro.get('selic', 0), "Cost of Capital", False),
    ("IPCA (12m)", latest_macro.get('ipca_12m', 0), "Inflation", True), # High inflation usually bad (-), but color depends
    ("IBC-Br (YoY)", latest_macro.get('ibc_br_yoy', 0), "Economic Activity", True),
    ("PIB (YoY)", latest_macro.get('pib_yoy', 0), "GDP Growth", True),
    ("USD/BRL", latest_macro.get('usd_brl', 0), "Exchange Rate", False),
    ("Real Rates", latest_macro.get('juro_real', 0), "Selic - IPCA", True),
]

html_rows = ""
for name, val, ctx, is_growth in ledger_items:
    # Color logic: 
    # Growth > 0 is good (Green). Inflation > Target (say 4.5%) is Bad (Red)? 
    # Simplify: Positive numbers green, negative red? No.
    # Selic high = Red (contractionary). Selic low = Green?
    # Contextual coloring requires nuance.
    # For "Ledger", let's stick to simple accounting style (Debit/Credit look).
    val_class = "val-neu"
    if is_growth:
        val_class = "val-pos" if val > 0 else "val-neg"
    
    val_str = fmt_val(val, is_pct=(name != "USD/BRL"))
    html_rows += f"<tr><td>{name}</td><td class='{val_class}'>{val_str}</td><td>{ctx}</td></tr>"

st.markdown(f"""
<table class="ledger-table">
    <thead><tr><th>Indicator</th><th>Value</th><th>Context</th></tr></thead>
    <tbody>{html_rows}</tbody>
</table>
""", unsafe_allow_html=True)


# --- MACRO TRENDS ---
st.subheader("📈 Digital Garden: Macro Trends")

col_trend1, col_trend2 = st.columns(2)

with col_trend1:
    st.markdown("**Activity: GDP & IBC-Br**")
    fig_act = go.Figure()
    # Handle NaN in activity
    act_df = macro_df[['ibc_br_yoy', 'pib_yoy']].dropna()
    act_df = act_df.loc[act_df.index >= "2018-01-01"] # Zoom in
    
    fig_act.add_trace(go.Scatter(x=act_df.index, y=act_df['ibc_br_yoy'], name='IBC-Br (YoY)', line=dict(color='#205EA6')))
    fig_act.add_trace(go.Scatter(x=act_df.index, y=act_df['pib_yoy'], name='PIB (YoY)', line=dict(color='#879A39', dash='dot')))
    fig_act = make_flexoki_chart(fig_act)
    fig_act.update_layout(height=300, showlegend=True, legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig_act, use_container_width=True)

with col_trend2:
    st.markdown("**Rates: Selic vs IPCA**")
    fig_rates = go.Figure()
    rates_df = macro_df[['selic', 'ipca_12m', 'juro_real']].dropna()
    rates_df = rates_df.loc[rates_df.index >= "2018-01-01"]
    
    fig_rates.add_trace(go.Scatter(x=rates_df.index, y=rates_df['selic'], name='Selic', line=dict(color='#D14D41')))
    fig_rates.add_trace(go.Scatter(x=rates_df.index, y=rates_df['ipca_12m'], name='IPCA 12m', line=dict(color='#100F0F', dash='dot')))
    fig_rates.add_trace(go.Scatter(x=rates_df.index, y=rates_df['juro_real'], name='Real Rate', line=dict(color='#879A39'), fill='tozeroy', opacity=0.1))
    
    fig_rates = make_flexoki_chart(fig_rates)
    fig_rates.update_layout(height=300, showlegend=True, legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig_rates, use_container_width=True)


# --- CURRENT REGIME DISPLAY ---
st.divider()

current_style = get_regime_style(result.current_regime)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Current Regime",
        f"{current_style['icon']} {result.current_regime.value}",
    )

with col2:
    # Get probability of current regime
    prob_col = f"P({result.current_regime.value})"
    if prob_col in result.regime_probabilities.columns:
        confidence = result.regime_probabilities[prob_col].iloc[-1]
    else:
        confidence = 0.5
    st.metric("Confidence", f"{confidence:.1%}")

with col3:
    # Days in current regime
    regime_history = result.regime_history
    current = regime_history.iloc[-1]
    streak = 0
    for val in reversed(regime_history.values):
        if val == current:
            streak += 1
        else:
            break
    st.metric("Days in Regime", streak)

with col4:
    st.metric("Suggested Strategy", current_style['strategy'][:20])

# --- REGIME TIMELINE ---
st.divider()
st.subheader("📅 Regime Timeline")

# Create regime timeline chart
fig_timeline = go.Figure()

# Map regimes to colors
regime_colors = {
    'Bull': '#879A39',
    'Bear': '#D14D41', 
    'Sideways': '#205EA6',
}

# Get market return for overlay
market_return = returns.mean(axis=1)
cumulative_return = (1 + market_return).cumprod() - 1

# Align with regime history
aligned_dates = result.regime_history.index.intersection(cumulative_return.index)
aligned_return = cumulative_return.loc[aligned_dates]
aligned_regime = result.regime_history.loc[aligned_dates]

# Add cumulative return line
fig_timeline.add_trace(go.Scatter(
    x=aligned_dates,
    y=aligned_return * 100,  # Convert to %
    mode='lines',
    name='Market Return',
    line=dict(color='#100F0F', width=2)
))

# Add regime background (using shapes)
regime_starts = []
current_regime = None
start_date = None

for date, regime in aligned_regime.items():
    if regime != current_regime:
        if current_regime is not None:
            regime_starts.append((start_date, date, current_regime))
        current_regime = regime
        start_date = date

# Don't forget the last regime
if current_regime is not None:
    regime_starts.append((start_date, aligned_dates[-1], current_regime))

# Add colored rectangles for each regime period
shapes = []
for start, end, regime in regime_starts:
    color = regime_colors.get(regime, '#E6E4D9')
    shapes.append(dict(
        type="rect",
        x0=start, x1=end,
        y0=0, y1=1,
        xref="x", yref="paper",
        fillcolor=color,
        opacity=0.15,
        layer="below",
        line_width=0,
    ))

fig_timeline.update_layout(
    shapes=shapes,
    title="Cumulative Market Return with Regime Overlay",
    xaxis_title="Date",
    yaxis_title="Cumulative Return (%)",
    showlegend=True,
    legend=dict(orientation="h", y=1.1),
)

fig_timeline = make_flexoki_chart(fig_timeline)
fig_timeline.update_layout(height=400)

st.plotly_chart(fig_timeline, use_container_width=True)

# Legend
col_legend = st.columns(3)
for i, (regime, color) in enumerate(regime_colors.items()):
    with col_legend[i]:
        st.markdown(f"<span style='color:{color}'>●</span> **{regime}**", unsafe_allow_html=True)

# --- TRANSITION MATRIX ---
st.divider()

col_matrix, col_stats = st.columns([1, 1])

with col_matrix:
    st.subheader("🔄 Transition Probabilities")
    
    # Create transition matrix heatmap
    trans_matrix = result.transition_matrix
    regime_names = [MarketRegime.BEAR.value, MarketRegime.SIDEWAYS.value, MarketRegime.BULL.value][:n_regimes]
    
    fig_trans = px.imshow(
        trans_matrix,
        x=regime_names,
        y=regime_names,
        color_continuous_scale=[[0, '#FFFCF0'], [1, '#205EA6']],
        text_auto='.2f',
        labels={'x': 'To', 'y': 'From', 'color': 'P(transition)'}
    )
    fig_trans.update_layout(
        title="P(Regime[t+1] | Regime[t])",
        height=350,
    )
    fig_trans = make_flexoki_chart(fig_trans)
    st.plotly_chart(fig_trans, use_container_width=True)

with col_stats:
    st.subheader("📈 Regime Statistics")
    
    # Regime counts
    regime_counts = result.regime_history.value_counts()
    total_days = len(result.regime_history)
    
    stats_data = []
    for i in range(n_regimes):
        regime = regime_names[i] if i < len(regime_names) else f"Regime {i}"
        count = regime_counts.get(regime, 0)
        pct = count / total_days if total_days > 0 else 0
        mean_ret = result.regime_means[i] if i < len(result.regime_means) else 0
        vol = result.regime_volatilities[i] if i < len(result.regime_volatilities) else 0
        
        stats_data.append({
            'Regime': regime,
            'Days': count,
            'Frequency': f"{pct:.1%}",
            'Avg Return': f"{mean_ret*100:.2f}%",
            'Volatility': f"{vol*100:.2f}%",
        })
    
    stats_df = pd.DataFrame(stats_data)
    st.dataframe(stats_df, use_container_width=True, hide_index=True)

# --- PROBABILITY TIMESERIES ---
st.divider()
st.subheader("📊 Regime Probabilities Over Time")

fig_probs = go.Figure()

for col in result.regime_probabilities.columns:
    regime_name = col.replace('P(', '').replace(')', '')
    color = regime_colors.get(regime_name, '#100F0F')
    
    fig_probs.add_trace(go.Scatter(
        x=result.regime_probabilities.index,
        y=result.regime_probabilities[col],
        mode='lines',
        name=col,
        line=dict(color=color, width=1.5),
        fill='tozeroy',
        # Robust hex to rgba conversion
        fillcolor=f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.1)" if color.startswith('#') else color,
    ))

fig_probs.update_layout(
    title="P(Regime) at Each Time Step",
    xaxis_title="Date",
    yaxis_title="Probability",
    yaxis=dict(range=[0, 1]),
    showlegend=True,
    legend=dict(orientation="h", y=1.1),
)
fig_probs = make_flexoki_chart(fig_probs)
fig_probs.update_layout(height=350)

st.plotly_chart(fig_probs, use_container_width=True)

# Interpretation
st.caption("""
💡 **Interpretation:** 
- **High certainty** = one regime dominates (probability near 1.0)
- **Mixed probabilities** = market in transition state
- Long stretches in one regime indicate stable market conditions
""")
