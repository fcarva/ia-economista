# -*- coding: utf-8 -*-
import streamlit as st
import plotly.graph_objects as go
import plotly.io as pio

def load_css():
    """
    Injects Flexoki Theme with enhanced Financial UI/UX.
    Concept: 'The Digital Ledger' - High readability, warm background, ink-like data.
    """
    st.markdown("""
        <style>
        /* 1. TYPOGRAPHY & FONTS */
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,300;400;500;600&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap');

        :root {
            /* FLEXOKI LIGHT PALETTE */
            --paper: #FFFCF0;
            --ink: #100F0F;
            --ink-subtle: #6F6E69;  /* Base-600 */
            --border: #E6E4D9;      /* Base-200 */
            --card-bg: #F2F0E5;     /* Base-50 - Slightly darker than paper for contrast */
            
            /* SEMANTIC FINANCIAL COLORS (Flexoki) */
            --profit: #66800B;      /* Green-600 */
            --loss: #AF3029;        /* Red-600 */
            --action: #205EA6;      /* Blue-600 */
            --alert: #AD8301;       /* Yellow-600 */
            --purple: #5E409D;      /* Purple-600 (for secondary data) */

            /* UI VARS */
            --radius: 6px;
            --shadow: 0 1px 2px rgba(16, 15, 15, 0.05);
        }

        /* 2. BASE LAYOUT */
        .stApp {
            background-color: var(--paper);
            color: var(--ink);
        }

        /* 3. TYPOGRAPHY HIERARCHY */
        h1, h2, h3 {
            font-family: 'Fraunces', serif !important;
            font-weight: 500 !important;
            letter-spacing: -0.03em !important;
            color: var(--ink);
        }
        
        h1 { font-size: 2.5rem !important; border-bottom: 2px solid var(--border); padding-bottom: 0.5rem; }
        h2 { font-size: 1.8rem !important; margin-top: 2rem !important; }
        h3 { font-size: 1.3rem !important; color: var(--ink-subtle) !important; font-family: 'Inter', sans-serif !important; text-transform: uppercase; letter-spacing: 0.05em; }

        p, li, .stMarkdown {
            font-family: 'Inter', sans-serif;
            line-height: 1.6;
            color: #3B3A37; /* Base-800 for softer reading */
        }

        /* 4. METRIC CARDS (The "Index Card" Look) */
        div[data-testid="stMetric"] {
            background-color: #FFFFFF;
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 16px;
            box-shadow: var(--shadow);
            transition: transform 0.2s;
        }
        
        div[data-testid="stMetric"]:hover {
            border-color: var(--action);
            transform: translateY(-2px);
        }

        label[data-testid="stMetricLabel"] {
            font-family: 'Inter', sans-serif;
            font-size: 0.85rem;
            color: var(--ink-subtle);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        div[data-testid="stMetricValue"] {
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.6rem !important;
            color: var(--ink);
            font-weight: 600;
        }

        /* Delta Colors Override */
        div[data-testid="stMetricDelta"] svg { display: none; } /* Hide default arrows if desired, or style them */
        div[data-testid="stMetricDelta"] > div {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.8rem;
            background-color: var(--paper);
            padding: 2px 6px;
            border-radius: 4px;
            border: 1px solid var(--border);
        }

        /* 5. DATAFRAMES (The "Ledger" Look) */
        div[data-testid="stDataFrame"] {
            border-radius: 6px;
            overflow: hidden;
        }

        /* 6. SIDEBAR (Subtle Integration) */
        section[data-testid="stSidebar"] {
            background-color: var(--sidebar-bg);
            border-right: 1px solid var(--border-color);
        }
        
        section[data-testid="stSidebar"] h1, 
        section[data-testid="stSidebar"] h2, 
        section[data-testid="stSidebar"] h3 {
            font-family: 'Inter', sans-serif !important; /* Cleaner sidebar headers */
            font-weight: 600;
        }

        /* 7. CONTAINERS & CARDS */
        div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"] {
            /* Logic to target inner containers if needed */
        }
        
        /* Custom border for st.container(border=True) */
        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-color: var(--border-color) !important;
            background-color: #fffcf0;
            border-radius: 8px;
        }

        /* 8. ALERTS & CALLOUTS */
        .stAlert {
            background-color: var(--sidebar-bg);
            border: 1px solid var(--border-color);
            font-family: 'Inter', sans-serif;
        }

        /* 9. BUTTONS (Minimalist) */
        .stButton button {
            background-color: var(--bg-color);
            border: 1px solid var(--border-color);
            color: var(--text-color);
            font-family: 'Inter', sans-serif;
            font-weight: 500;
            border-radius: 6px;
            transition: all 0.2s;
        }

        .stButton button:hover {
            border-color: var(--accent-color);
            color: var(--accent-color);
            background-color: #fff;
        }
        
        /* Primary Button Override */
        .stButton button[kind="primary"] {
            background-color: var(--accent-color);
            color: #fff;
            border: none;
        }

        </style>
    """, unsafe_allow_html=True)

def make_flexoki_chart(fig):
    """Applies Flexoki theme to a Plotly figure."""
    fig.update_layout(
        paper_bgcolor="#FFFCF0", # Match page background
        plot_bgcolor="#FFFCF0",  # Match page background
        font={'family': "JetBrains Mono", 'color': "#100F0F"},
        xaxis=dict(
            gridcolor="#E6E4D9", 
            zerolinecolor="#E6E4D9",
            showline=True,
            linewidth=1,
            linecolor="#100F0F"
        ),
        yaxis=dict(
            gridcolor="#E6E4D9", 
            zerolinecolor="#E6E4D9",
            showline=True,
            linewidth=1,
            linecolor="#100F0F"
        ),
        hovermode="x unified",
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(
            bgcolor="rgba(255,255,255,0.5)",
            bordercolor="#E6E4D9",
            borderwidth=1
        )
    )
    return fig
