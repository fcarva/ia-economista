"""
Causal Network Topology (Gitcoin-Style Force-Directed Graph)
=============================================================

Interactive visualization of lead-lag causal relationships.
Uses streamlit-agraph for physics-based node clustering.

Track 5 Enhancement: Bloomberg Terminal UX
"""

import streamlit as st
import networkx as nx
import pandas as pd
import numpy as np
from streamlit_agraph import agraph, Node, Edge, Config
from cointegration_gnn.data.data_loader import DataLoader
from cointegration_gnn.config import default_config
from dashboard.utils import load_css

# Page Config
st.set_page_config(page_title="Causal Topology", page_icon="🕸️", layout="wide")
load_css()

st.title("🕸️ Causal Topology")
st.markdown("### Lead-Lag Influence Structure (Research View)")

# Sidebar Controls
with st.sidebar:
    st.header("⚙️ Graph Physics")
    # Tweak default threshold to be meaningful for daily returns
    threshold = st.slider("Correlation Threshold (Lead-Lag)", 0.0, 0.5, 0.15, step=0.01)
    physics_gravity = st.slider("Gravity (Clustering)", -5000, -100, -1200)
    node_size_scale = st.slider("Node Size Scale", 10, 100, 30)
    
    st.divider()
    st.info("ℹ️ **Algorithms**: Pairwise Cross-Correlation at Lag 1. If Corr(A_t-1, B_t) > Threshold, then A -> B.")

# --- 1. LOAD DATA & CALCULATE LEAD-LAG ---
@st.cache_data
def build_lead_lag_network(threshold_val):
    """Build causal network from Lead-Lag Cross-Correlation."""
    # Load data
    loader = DataLoader(tickers=default_config.data.tickers)
    prices, returns = loader.fetch_data(
        start=default_config.data.train_start, 
        end=default_config.data.train_end
    )

    if returns.empty:
        return nx.DiGraph()
    
    # 1. Calculate Cross-Correlation Lag Matrix
    tickers = default_config.data.tickers
    n = len(tickers)
    
    # Adjacency Matrix
    adj = pd.DataFrame(0.0, index=tickers, columns=tickers)
    
    # Vectorized or Iterative correlation
    # For A leading B: Corr(A.shift(1), B)
    lagged_returns = returns.shift(1)
    
    # Using pairwise correlation between Lagged(A) and Current(B)
    # This matrix: Index=Lagged(Source), Columns=Current(Target)
    # corr_matrix[A, B] = Corr(A_t-1, B_t)
    
    # We can compute this efficiently:
    # Concatenate lagged and current, compute corr, extract relevant quadrant
    # Or just loop for clarity (N is small ~20)
    
    for source in tickers:
        src_series = returns[source].shift(1).iloc[1:] # t-1
        for target in tickers:
            if source == target:
                continue
            tgt_series = returns[target].iloc[1:] # t
            
            # Compute correlation
            corr = src_series.corr(tgt_series)
            
            if corr > threshold_val:
                adj.loc[source, target] = corr

    # Create NetworkX Graph
    G = nx.DiGraph()
    
    # Sectors Metadata
    sectors = {
        'PETR4.SA': 'Energy', 'VALE3.SA': 'Materials', 'ITUB4.SA': 'Financials', 
        'BBDC4.SA': 'Financials', 'BBAS3.SA': 'Financials', 'WEGE3.SA': 'Industrial',
        'PRIO3.SA': 'Energy', 'RENT3.SA': 'Industrial', 'JBSS3.SA': 'Consumer', 
        'ELET3.SA': 'Utilities',
        # US ETFs
        'XLK': 'Technology', 'XLF': 'Financials', 'XLE': 'Energy',
        'XLV': 'Healthcare', 'XLI': 'Industrial', 'XLP': 'Consumer',
        'XLB': 'Materials', 'XLY': 'Consumer', 'XLU': 'Utilities'
    }
    
    # Flexoki Sector Colors
    colors = {
        'Energy': '#D14D41',      # Red
        'Materials': '#DA702C',   # Orange
        'Financials': '#205EA6',  # Blue
        'Industrial': '#24837B',  # Teal
        'Consumer': '#879A39',    # Green
        'Utilities': '#5F554F',   # Grey
        'Technology': '#8B7EC8',  # Purple
        'Healthcare': '#CE5D97',  # Pink
        'Unknown': '#100F0F'      # Black
    }

    # Add Nodes
    for ticker in tickers:
        sector = sectors.get(ticker, 'Unknown')
        G.add_node(ticker, group=sector, color=colors.get(sector, '#100F0F'))
    
    # Add Edges
    for source in tickers:
        for target in tickers:
            weight = adj.loc[source, target]
            if weight > 0:
                G.add_edge(source, target, weight=weight)
        
    return G

G = build_lead_lag_network(threshold)

# --- 2. CONVERT TO AGRAPH (The "Gitcoin" Look) ---
nodes = []
edges = []

# Calculate Centrality for Sizing
# Out-Degree checks "Influence" (How many do I lead?)
# In-Degree checks "Sensitivity" (How many lead me?)
if len(G.nodes) > 0:
    out_degree = dict(G.out_degree(weight='weight'))
    in_degree = dict(G.in_degree(weight='weight'))
    
    # Normalize for visual size
    max_out = max(out_degree.values()) if out_degree else 1
    max_out = max(max_out, 0.001) # Avoid div/0

    for node_id in G.nodes:
        sector = G.nodes[node_id].get('group', 'Unknown')
        color = G.nodes[node_id].get('color', '#100F0F')
        
        # Influence Score
        influence = out_degree.get(node_id, 0)
        
        # Size logic: Base + Influence
        size = 15 + (influence / max_out * node_size_scale)
        
        # Label: Ticker
        label = node_id.replace('.SA', '')
        
        nodes.append(Node(
            id=node_id,
            label=label,
            size=size,
            color=color,
            font={'color': '#100F0F', 'face': 'Inter'},
            title=f"Influence Score: {influence:.2f}" # Tooltip
        ))

    for u, v, d in G.edges(data=True):
        weight = d.get('weight', 0.1)
        # Width proportional to correlation
        width = max(0.5, weight * 10)
        
        edges.append(Edge(
            source=u,
            target=v,
            color="#E6E4D9",  # Subtle edges
            width=width,
            type="CURVE_SMOOTH" 
        ))

# --- 3. PHYSICS CONFIGURATION ---
config = Config(
    width=900,
    height=600,
    directed=True, 
    physics=True, 
    hierarchical=False,
    nodeHighlightBehavior=True,
    highlightColor="#F7A7A6",
    collapsible=False,
    node={'labelProperty': 'label'},
    link={'labelProperty': 'label', 'renderLabel': False},
    # Physics engine properties
    gravity=physics_gravity, 
    minVelocity=0.75,
    solver='forceAtlas2Based', # Good for Gitcoin/Organic style
    stabilization=False
)

# --- 4. RENDER & METRICS ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Nodes", len(nodes))
col2.metric("Lead-Lag Edges", len(edges))
col3.metric("Network Density", f"{nx.density(G):.2%}" if len(G.nodes) > 0 else "0%")

# Find Top Influencer
if len(G.nodes) > 0:
    top_influencer = max(dict(G.out_degree(weight='weight')), key=dict(G.out_degree(weight='weight')).get)
    col4.metric("Top Influencer (Source)", top_influencer.replace('.SA', ''))
else:
    col4.metric("Top Influencer (Source)", "N/A")

st.divider()

if len(nodes) > 0:
    return_value = agraph(nodes=nodes, edges=edges, config=config)
    
    if return_value:
        st.info(f"📍 Analysis for: **{return_value}**")
        
        # Show connections for selected node
        if return_value in G.nodes:
            # Predecessors (Who leads this asset?)
            preds = sorted(list(G.predecessors(return_value)))
            # Successors (Who does this asset lead?)
            succs = sorted(list(G.successors(return_value)))
            
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**⬅️ Lagging Behind:**")
                st.caption(f"Assets that move BEFORE {return_value} (Leaders)")
                if preds:
                    for p in preds:
                        w = G.edges[p, return_value]['weight']
                        st.markdown(f"- {p} (Corr: {w:.2f})")
                else:
                    st.write("None (Independent Mover)")
                    
            with c2:
                st.markdown(f"**➡️ Leading:**")
                st.caption(f"Assets that move AFTER {return_value} (Followers)")
                if succs:
                    for s in succs:
                        w = G.edges[return_value, s]['weight']
                        st.markdown(f"- {s} (Corr: {w:.2f})")
                else:
                    st.write("None")
else:
    st.warning("No significant lead-lag relationships found. Try lowering the threshold.")

# Explanation
st.divider()
with st.expander("ℹ️ How to interpret this graph?"):
    st.markdown("""
    **Gitcoin-Style Causal Network:**
    - **Nodes**: Assets in the universe. **Size** represents "Influence Power" (how many other assets follow its moves).
    - **Edges**: Directed arrows A → B mean that **A's return today correlates with B's return tomorrow**.
    - **Clusters**: Assets that move together or influence each other will cluster naturally.
    - **Strategy**: Look for large nodes (Leading Indicators) to predict movements in connected smaller nodes.
    """)
