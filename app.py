import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

# Index spot prices
indices = {
    "NIFTY": 22500,
    "BANKNIFTY": 48200,
    "FINNIFTY": 21500
}

st.set_page_config(page_title="Options Strategy Dashboard", layout="wide")
st.title("📊 Options Strategy Dashboard - Vega + Theta Analysis")

# User inputs
index = st.selectbox("Select Index", list(indices.keys()))
spot_price = indices[index]

col1, col2 = st.columns(2)
with col1:
    strategy = st.selectbox("Strategy", [
        "Call Straddle", "Put Straddle",
        "Strangle", "Bull Put Spread"
    ])
with col2:
    expiry_days = st.slider("Days to Expiry", 1, 30, 7)

volatility = st.slider("Implied Volatility (%)", 10, 100, 20) / 100
risk_free_rate = 0.05
lot_size = st.number_input("Lot Size", 25, 1000, 50)

# Black-Scholes pricing & Greeks
def black_scholes_greeks(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) -
                 r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) +
                 r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365

    vega = S * norm.pdf(d1) * np.sqrt(T) / 100
    return price, vega, theta

# Simulate P&L, Vega, Theta
def simulate_strategy():
    prices = np.linspace(spot_price * 0.8, spot_price * 1.2, 100)
    T = expiry_days / 365
    atm = round(spot_price / 50) * 50
    otm_c = atm + 100
    otm_p = atm - 100

    payoff, vega_list, theta_list = [], [], []

    for S in prices:
        pnl = vega = theta = 0

        if strategy == "Call Straddle":
            c_price, c_vega, c_theta = black_scholes_greeks(S, atm, T, risk_free_rate, volatility, 'call')
            c_init = black_scholes_greeks(spot_price, atm, T, risk_free_rate, volatility, 'call')[0]
            pnl = (c_price - c_init) * -2
            vega = c_vega * 2
            theta = c_theta * 2

        elif strategy == "Put Straddle":
            p_price, p_vega, p_theta = black_scholes_greeks(S, atm, T, risk_free_rate, volatility, 'put')
            p_init = black_scholes_greeks(spot_price, atm, T, risk_free_rate, volatility, 'put')[0]
            pnl = (p_price - p_init) * -2
            vega = p_vega * 2
            theta = p_theta * 2

        elif strategy == "Strangle":
            c_price, c_vega, c_theta = black_scholes_greeks(S, otm_c, T, risk_free_rate, volatility, 'call')
            p_price, p_vega, p_theta = black_scholes_greeks(S, otm_p, T, risk_free_rate, volatility, 'put')
            c_init = black_scholes_greeks(spot_price, otm_c, T, risk_free_rate, volatility, 'call')[0]
            p_init = black_scholes_greeks(spot_price, otm_p, T, risk_free_rate, volatility, 'put')[0]
            pnl = (c_price + p_price - c_init - p_init) * -1
            vega = c_vega + p_vega
            theta = c_theta + p_theta

        elif strategy == "Bull Put Spread":
            short_strike = atm
            long_strike = atm - 100
            short_p, short_v, short_t = black_scholes_greeks(S, short_strike, T, risk_free_rate, volatility, 'put')
            long_p, long_v, long_t = black_scholes_greeks(S, long_strike, T, risk_free_rate, volatility, 'put')
            short_init = black_scholes_greeks(spot_price, short_strike, T, risk_free_rate, volatility, 'put')[0]
            long_init = black_scholes_greeks(spot_price, long_strike, T, risk_free_rate, volatility, 'put')[0]
            pnl = (short_p - long_p - (short_init - long_init))
            vega = short_v - long_v
            theta = short_t - long_t

        payoff.append(pnl * lot_size)
        vega_list.append(vega * lot_size)
        theta_list.append(theta * lot_size)

    return prices, payoff, vega_list, theta_list

# Generate and display results
prices, payoff, vegas, thetas = simulate_strategy()
df = pd.DataFrame({
    "Price": prices,
    "Payoff": payoff,
    "Vega": vegas,
    "Theta": thetas
})

fig, ax = plt.subplots()
ax.plot(prices, payoff, label="Payoff", color='blue')
ax.plot(prices, vegas, label="Vega", linestyle="--", color='orange')
ax.plot(prices, thetas, label="Theta", linestyle="--", color='green')
ax.axhline(0, color='black', linewidth=0.5)
ax.set_xlabel("Underlying Price")
ax.set_ylabel("Value")
ax.set_title(f"{strategy} | {index}")
ax.legend()
st.pyplot(fig)

# Export CSV
st.download_button("📤 Export to CSV", df.to_csv(index=False), "strategy_output.csv", "text/csv")

# Backtesting placeholder
st.header("🔍 Strategy Backtesting (Coming Soon)")
st.info("This module will simulate historical performance of selected strategy using past data.")
