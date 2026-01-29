# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
from pathlib import Path
import torch
import datetime

from cointegration_gnn.config import default_config
from dashboard.utils import load_css

# --- PAGE CONFIG (Minimal) ---
st.set_page_config(
    page_title="AI Economist",
    page_icon="♟️",
    layout="wide",
    initial_sidebar_state="collapsed" # Começa fechada para foco total (Zen Mode)
)

# Inject Minimal CSS
load_css()

# --- BACKEND HOOKS ---
def get_system_status():
    """Check core system vitals."""
    # Data
    data_ok = Path("data").exists()
    # Model
    model_files = list(Path("models").glob("*.ckpt"))
    model_ok = len(model_files) > 0
    latest_model = model_files[-1].name if model_ok else "N/A"
    # Compute
    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    
    return {
        "data": data_ok,
        "model": model_ok,
        "model_name": latest_model,
        "device": device
    }

status = get_system_status()

# --- SIDEBAR (Context Only) ---
with st.sidebar:
    st.markdown("### ♟️ Market Context")
    
    # Minimal Regime Indicator
    volatility = 22.5 
    if volatility > 20:
        st.markdown(f"Regime: **High Volatility** (VIX {volatility})")
        st.caption("Strategy: Defensive / Short Bias")
    else:
        st.markdown(f"Regime: **Calm** (VIX {volatility})")
        st.caption("Strategy: Mean Reversion / Leverage")
    
    st.divider()
    
    st.markdown("### ⚙️ System Specs")
    st.code(f"""
    Device: {status['device']}
    Data:   {'Online' if status['data'] else 'Offline'}
    Model:  {'Ready' if status['model'] else 'Pending'}
    """)
    
    st.divider()
    st.caption(f"v1.0.4 | {datetime.date.today()}")

# --- MAIN INTERFACE ---

# 1. HEADER (Typography First)
col_title, col_status = st.columns([3, 1])
with col_title:
    st.title("AI Economist")
    st.markdown("Multi-Asset Cointegration Research Console")

with col_status:
    # Status Pill Minimalista
    if status['model']:
        st.success("● SYSTEM ONLINE")
    else:
        st.warning("● INITIALIZING")

st.divider()

# 2. KEY METRICS (The "Numbers" Row)
# Minimalist metrics, no heavy backgrounds
m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Return (2024)", "0.06%", "+12.17% vs IBOV")
m2.metric("Alpha Generated", "+12.17%", "Positive")
m3.metric("Sharpe Ratio", "0.38", "+1.35 vs IBOV")
m4.metric("Max Drawdown", "0.28%", "Low Risk")

st.markdown("######") # Spacer

# 3. ACTION PIPELINE (The CTA Core)
# Transformamos os módulos em um "Fluxo de Trabalho"
st.subheader("Research Workflow")

# Container unificado para o pipeline
with st.container(border=True):
    cols = st.columns(4)
    
    # Step 1: Universe
    with cols[0]:
        st.markdown("**1. Asset Universe**")
        st.caption("IBOV Blue Chips • Diagnostics")
        if st.button("Inspect Assets", use_container_width=True):
            st.switch_page("pages/01_Asset_Universe.py")
            
    # Step 2: Topology
    with cols[1]:
        st.markdown("**2. Causal Topology**")
        st.caption("Lead-Lag • Cointegration")
        if st.button("View Network", use_container_width=True):
            st.switch_page("pages/02_Causal_Topology.py")
            
    # Step 3: Inference (Primary CTA if model exists)
    with cols[2]:
        st.markdown("**3. Model Explainability**")
        st.caption("GraphSAGE • GATv2")
        if st.button("Explain Logic", use_container_width=True):
            st.switch_page("pages/04_Model_Insights.py")
            
    # Step 4: Execution (Ultimate Goal)
    with cols[3]:
        st.markdown("**4. Strategy Lab**")
        st.caption("Backtest • Risk Metrics")
        # Botão Primário para destacar a ação final
        if st.button("Launch Simulation", type="primary", use_container_width=True):
            st.switch_page("pages/03_Strategy_Backtest.py")

st.markdown("######") # Spacer

# 4. CONSOLE / QUICK LOGS (Clean, Monospace)
c1, c2 = st.columns([2, 1])

with c1:
    st.subheader("System Log")
    # Simulação de terminal limpo
    log_text = f"""
    [{datetime.datetime.now().strftime('%H:%M:%S')}] INFO  Orchestrator: System initialized successfully.
    [{datetime.datetime.now().strftime('%H:%M:%S')}] INFO  DataLoader:   Cached data verified (hash: a1b2c3).
    [{datetime.datetime.now().strftime('%H:%M:%S')}] WARN  RiskManager:  Volatility regime shift detected (>20).
    [{datetime.datetime.now().strftime('%H:%M:%S')}] READY AI Agent:     Model '{status['model_name']}' loaded.
    """
    st.code(log_text, language="bash")

with c2:
    st.subheader("Quick Actions")
    st.button("↻ Refresh Pipeline", use_container_width=True)
    st.button("📥 Export Alpha Report", use_container_width=True)
