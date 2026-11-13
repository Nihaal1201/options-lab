"""
Options Lab dashboard (Module 7).

Run on your Mac:   streamlit run app.py
It opens in your browser. Anyone can use it once it's deployed online.

Tabs:
  1. Volatility surface : live 3D implied vol surface for any US ticker
  2. Smile              : the smile for one expiry, puts and calls
  3. Term structure     : at-the-money vol and crash premium across expiries
  4. Option calculator  : price any option 3 ways + Greeks (no internet needed)

If Yahoo Finance can't be reached, the app falls back to the most recent
snapshot saved in data/ by scripts/module6_vol_surface.py.
"""

import glob
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import market_data as md
from src import vol_surface as vs
from src.binomial import binomial_price
from src.black_scholes import price as bs_price
from src.greeks import all_greeks
from src.monte_carlo import mc_european

st.set_page_config(page_title="Options Lab", page_icon="📈", layout="wide")


# ---------------------------------------------------------------- data ----
@st.cache_data(ttl=900, show_spinner=False)   # re-download at most every 15 minutes
def load_live(symbol, div_yield, n_expiries, max_days):
    ticker = md.get_ticker(symbol)
    S = md.get_spot(ticker)
    r = md.get_risk_free_rate()
    expiries = vs.choose_expiries(ticker.options, min_days=7, max_days=max_days, n=n_expiries)
    surface = vs.download_surface_data(ticker, S, r, div_yield, expiries, pause=0.2)
    if surface.empty:
        raise ValueError("no usable options returned")
    rv = md.realized_vol(ticker, "1mo")
    return surface, S, r, rv


def load_snapshot(symbol):
    files = sorted(glob.glob(f"data/{symbol}_surface_*.csv"))
    if not files:
        return None, None
    return pd.read_csv(files[-1]), files[-1]


# ------------------------------------------------------------- sidebar ----
with st.sidebar:
    st.title("Options Lab")
    st.caption("Option pricing and implied volatility, built from scratch in Python.")
    with st.form("controls"):
        symbol = st.text_input("US ticker", value="SPY").strip().upper()
        div_yield = st.number_input("Dividend yield", value=0.012, min_value=0.0,
                                    max_value=0.10, step=0.001, format="%.3f",
                                    help="SPY ~0.012, AAPL ~0.005, most growth stocks 0")
        n_expiries = st.slider("Expiries to download", 4, 15, 8)
        max_days = st.slider("Furthest expiry (days)", 60, 730, 365, step=30)
        st.form_submit_button("Load", type="primary")
    st.markdown(
        "**How to read this**\n"
        "- **Implied vol**: the market's forecast of how much the stock will swing.\n"
        "- **Skew**: crash protection (low strikes) costs more volatility.\n"
        "- **Term structure**: upward = calm, inverted = near-term fear."
    )

# ------------------------------------------------------------ load data ----
source_note = None
surface, S, r, rv = None, None, None, None
with st.spinner(f"Downloading {symbol} option chains from Yahoo Finance..."):
    try:
        surface, S, r, rv = load_live(symbol, div_yield, n_expiries, max_days)
        source_note = "Live data from Yahoo Finance (refreshed every 15 minutes)."
    except Exception as e:
        surface, path = load_snapshot(symbol)
        if surface is None:
            st.error(f"Couldn't download option data for **{symbol}** ({type(e).__name__}). "
                     "Check the ticker, or try again during US market hours. "
                     "The Option calculator tab still works.")
        else:
            S, r, rv = float(surface["spot"].iloc[0]), None, None
            source_note = (f"Live download failed ({type(e).__name__}: {str(e)[:150]}), "
                           f"so this shows the saved snapshot `{os.path.basename(path)}`.")

st.title(f"{symbol} implied volatility" if surface is not None else "Options Lab")
if source_note:
    st.caption(source_note)

# ------------------------------------------------- headline numbers ----
if surface is not None:
    ts = vs.term_structure(surface)
    cols = st.columns(5)
    cols[0].metric("Spot price", f"${S:,.2f}")
    if len(ts):
        cols[1].metric(f"ATM implied vol ({ts['days'].iloc[0]}d)", f"{ts['atm_iv'].iloc[0]:.1%}")
        cols[3].metric("Crash premium (90% strike)", f"{ts['skew_90_100'].mean():+.1%}",
                       help="Extra implied vol for a strike 10% below spot, averaged over expiries")
    if rv is not None:
        cols[2].metric("Realized vol (1 month)", f"{rv:.1%}")
    if len(ts) >= 2:
        upward = ts["atm_iv"].iloc[-1] > ts["atm_iv"].iloc[0]
        cols[4].metric("Term structure", "Upward" if upward else "Inverted",
                       help="Upward: longer expiries carry more vol (calm market). "
                            "Inverted: short-term vol is higher (near-term fear).")

tab_surface, tab_smile, tab_term, tab_calc = st.tabs(
    ["Volatility surface", "Smile", "Term structure", "Option calculator"])

# ---------------------------------------------------- market data tabs ----
if surface is not None:
    with tab_surface:
        M, D, Z = vs.interpolate_grid(surface)
        fig = go.Figure([
            go.Surface(x=M, y=D, z=Z * 100, colorscale="Viridis", opacity=0.92,
                       colorbar=dict(title="IV %"), name="Surface",
                       hovertemplate="Moneyness %{x:.2f}<br>%{y:.0f} days<br>IV %{z:.1f}%<extra></extra>"),
            go.Scatter3d(x=surface["moneyness"], y=surface["days"], z=surface["iv"] * 100,
                         mode="markers", marker=dict(size=2, color="black"), name="Market quotes",
                         hovertemplate="Moneyness %{x:.2f}<br>%{y:.0f} days<br>IV %{z:.1f}%<extra></extra>"),
        ])
        fig.update_layout(height=650, margin=dict(l=0, r=0, t=10, b=0),
                          scene=dict(xaxis_title="Moneyness (strike / spot)",
                                     yaxis_title="Days to expiry", zaxis_title="Implied vol (%)",
                                     camera=dict(eye=dict(x=-1.6, y=-1.6, z=0.8))))
        st.plotly_chart(fig, width="stretch")
        st.caption("Drag to rotate, scroll to zoom. Each black dot is one real option quote; "
                   "the surface is interpolated between them.")

    with tab_smile:
        expiries = sorted(surface["expiry"].unique())
        chosen = st.selectbox("Expiry", expiries, index=min(1, len(expiries) - 1))
        one = surface[surface["expiry"] == chosen].sort_values("strike")
        fig = go.Figure()
        for opt, color in [("put", "#1f77b4"), ("call", "#ff7f0e")]:
            sub = one[one["type"] == opt]
            fig.add_trace(go.Scatter(x=sub["strike"], y=sub["iv"] * 100, mode="lines+markers",
                                     name=f"OTM {opt}s", line=dict(color=color)))
        fig.add_vline(x=S, line_dash="dot", annotation_text=f"Spot ${S:,.0f}")
        if rv is not None:
            fig.add_hline(y=rv * 100, line_dash="dash",
                          annotation_text=f"1-month realized vol {rv:.1%}")
        fig.update_layout(height=480, xaxis_title="Strike", yaxis_title="Implied vol (%)",
                          margin=dict(t=30))
        st.plotly_chart(fig, width="stretch")
        st.dataframe(one[["type", "strike", "mid", "iv", "openInterest"]]
                     .rename(columns={"mid": "price", "iv": "implied vol", "openInterest": "open interest"})
                     .style.format({"price": "{:.2f}", "implied vol": "{:.1%}"}),
                     width="stretch", hide_index=True)

    with tab_term:
        if len(ts):
            fig = go.Figure([
                go.Scatter(x=ts["days"], y=ts["atm_iv"] * 100, mode="lines+markers",
                           name="At-the-money IV"),
                go.Scatter(x=ts["days"], y=(ts["atm_iv"] + ts["skew_90_100"]) * 100,
                           mode="lines+markers", name="IV at 90% strike", line=dict(dash="dash")),
            ])
            fig.update_layout(height=450, xaxis_title="Days to expiry", yaxis_title="Implied vol (%)",
                              margin=dict(t=30))
            st.plotly_chart(fig, width="stretch")
            st.dataframe(ts.rename(columns={"atm_iv": "ATM IV", "skew_90_100": "crash premium"})
                         .style.format({"ATM IV": "{:.1%}", "crash premium": "{:+.1%}"}),
                         width="stretch", hide_index=True)
        else:
            st.info("Not enough data to build a term structure for this ticker.")
else:
    for tab in (tab_surface, tab_smile, tab_term):
        with tab:
            st.info("No market data loaded. Try another ticker, or use the Option calculator tab.")

# -------------------------------------------------- option calculator ----
with tab_calc:
    st.subheader("Price any option three ways")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    default_S = float(round(S)) if surface is not None else 100.0
    calc_S = c1.number_input("Stock price", value=default_S, min_value=0.01)
    calc_K = c2.number_input("Strike", value=default_S, min_value=0.01)
    calc_days = c3.number_input("Days to expiry", value=30, min_value=1, max_value=3650)
    calc_r = c4.number_input("Interest rate", value=round(r, 4) if r else 0.04, step=0.005, format="%.4f")
    calc_vol = c5.number_input("Volatility", value=0.20, min_value=0.01, max_value=3.0, step=0.01)
    calc_type = c6.selectbox("Type", ["call", "put"])
    T = calc_days / 365

    bs = bs_price(calc_S, calc_K, T, calc_r, calc_vol, calc_type)
    american = binomial_price(calc_S, calc_K, T, calc_r, calc_vol, 500, calc_type, american=True)
    european_tree = binomial_price(calc_S, calc_K, T, calc_r, calc_vol, 500, calc_type)
    mc, se = mc_european(calc_S, calc_K, T, calc_r, calc_vol, 200_000, calc_type,
                         antithetic=True, seed=0)

    m1, m2, m3 = st.columns(3)
    m1.metric("Black-Scholes (European)", f"${bs:,.4f}")
    m2.metric("Binomial tree (American)", f"${american:,.4f}",
              delta=f"{american - european_tree:+.4f} early-exercise premium", delta_color="off")
    m3.metric("Monte Carlo (European)", f"${mc:,.4f}",
              delta=f"95% CI ±{1.96 * se:.4f}", delta_color="off")

    greeks = all_greeks(calc_S, calc_K, T, calc_r, calc_vol, calc_type)
    g = st.columns(5)
    labels = {"delta": "per $1 move", "gamma": "delta change per $1", "vega": "per 1 vol point",
              "theta": "per day", "rho": "per 1 rate point"}
    for col, (name, value) in zip(g, greeks.items()):
        col.metric(name.capitalize(), f"{value:+.4f}", help=labels[name])

    grid = np.linspace(calc_S * 0.6, calc_S * 1.4, 200)
    fig = go.Figure([
        go.Scatter(x=grid, y=bs_price(grid, calc_K, T, calc_r, calc_vol, calc_type),
                   name=f"Value today ({calc_days} days left)"),
        go.Scatter(x=grid, y=np.maximum(grid - calc_K, 0) if calc_type == "call"
                   else np.maximum(calc_K - grid, 0), name="Payoff at expiry",
                   line=dict(dash="dash", color="black")),
    ])
    fig.add_vline(x=calc_S, line_dash="dot")
    fig.update_layout(height=400, xaxis_title="Stock price", yaxis_title="Option value",
                      margin=dict(t=30))
    st.plotly_chart(fig, width="stretch")
