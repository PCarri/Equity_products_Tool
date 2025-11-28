
"""
conda install scipy
conda install numpy 
conda install pandas 
conda install matplotlib 
conda install seaborn 
conda install scikit-learn 
conda install jupyter
conda install spyder
conda install pytables
conda install ipython
conda install xlrd
pip install yfinance
pip install transformers
pip install streamlit
"""
import math #math already in python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import yfinance as yf
import streamlit as st
import matplotlib as mpl
mpl.rcParams['font.family'] = 'serif'
from scipy.integrate import quad


#
# Helper Functions
#
def dN(x):
	#Probability density function of standard normal random variable x.
	return math.exp(-0.5 * x ** 2) / math.sqrt(2 * math.pi)

def N(d):
#Cumulative density function of standard normal random variable x.
    return quad(lambda x: dN(x), -20, d, limit=50)[0]
 
def d1f(St, K, t, T, r, sigma):
#Black-Scholes-Merton d1 function.
#Parameters see e.g. BSM_call_value function.
    d1 = (math.log(St / K) + (r + 0.5 * sigma ** 2)
* (T - t)) / (sigma * math.sqrt(T - t))
    return d1

### Valuation Functions

def BSM_call_value(St, K, t, T, r, sigma):
    #Calculates Black-Scholes-Merton European call option value.
    """"
    Parameters
    ==========
    St: float
    stock/index level at time t
    K: float
    strike price
    t: float
    valuation date
    T: float
    date of maturity/time-to-maturity if t = 0; T > t
    r: float
    constant, risk-less short rate
    sigma: float
    volatility
    Returns
    =======4
    call_value: float
    European call present value at t
    """
    d1 = d1f(St, K, t, T, r, sigma)
    d2 = d1 - sigma * math.sqrt(T - t)
    call_value = St * N(d1) - math.exp(-r * (T - t)) * K * N(d2)
    return call_value

def BSM_put_value(St, K, t, T, r, sigma):
    ''' Calculates Black-Scholes-Merton European put option value.
    Parameters
    ==========
    St: float
    stock/index level at time t
    K: float
    strike price
    t: float
    valuation date
    T: float
    date of maturity/time-to-maturity if t = 0; T > t
    r: float
    constant, risk-less short rate
    sigma: float
    volatility
    Returns
    =======
    put_value: float
    European put present value at t
    '''
    put_value = BSM_call_value(St, K, t, T, r, sigma) \
    - St + math.exp(-r * (T - t)) * K
    return put_value

def zero_coupon_price(F, t, T, r):
    """
    Price of a zero-coupon bond with face value F, maturity T, continuous rate r.
    P(t) = F * exp(-r * (T - t))
    """
    return F * math.exp(-r * (T - t))

def simulate_gbm_paths(S0, T, r, sigma, n_steps, n_paths):
    """
    Simulate geometric Brownian motion paths.

    Parameters
    ----------
    S0 : float
        Initial stock price at t = 0.
    T : float
        Time horizon (in years).
    r : float
        Drift. If you want risk-neutral, use the risk-free rate.
    sigma : float
        Volatility.
    n_steps : int
        Number of time steps.
    n_paths : int
        Number of simulated paths.

    Returns
    -------
    time : np.ndarray, shape (n_steps + 1,)
        Time grid from 0 to T.
    paths : np.ndarray, shape (n_steps + 1, n_paths)
        Simulated price paths. paths[i, j] = price at time time[i] for path j.
    terminal_prices : np.ndarray, shape (n_paths,)
        Simulated prices at time T (last row of paths).
    """
    dt = T / n_steps  # time step

    # Normal increments: shape (n_steps, n_paths)
    Z = np.random.normal(0.0, 1.0, size=(n_steps, n_paths))

    # Log returns per step: (r - 0.5 sigma^2)*dt + sigma * sqrt(dt) * Z
    increments = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z

    # Cumulative sum over time (axis=0): log S_t
    log_S = np.log(S0) + np.cumsum(increments, axis=0)

    # Add initial log price row for t=0
    log_S0 = np.full((1, n_paths), np.log(S0))
    log_paths = np.vstack([log_S0, log_S])  # shape (n_steps + 1, n_paths)

    # Exponentiate to get prices
    paths = np.exp(log_paths)  # shape (n_steps + 1, n_paths)

    # Time grid
    time = np.linspace(0.0, T, n_steps + 1)

    # Terminal prices are last row
    terminal_prices = paths[-1, :]

    return time, paths, terminal_prices

# ---------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------
st.title("Equity-Linked Product Playground")
st.write(
    "Explore how the initial investment limit loses from an initial investment"
    "for capital protection and derivatives for equity participation."
)



t = 0.0  # valuation time always 0 in your setup


# 1. Investment amount & capital protection
investment_amount = st.slider(
    "Investment amount",
    min_value=1_000,
    max_value=1_0000,
    value=1_000,
    step=1_000,
)

investment_amount = float(investment_amount)

protection_pct = st.slider(
    "Capital protection at maturity (%)",
    min_value=0.0,
    max_value=100.0,
    value=100.0,
    step=10.0,
    format="%.0f%%",
)



st.markdown("### Step 1: Choose protection level and model parameters")

# 2. Model parameters (needed for ZCB and options)
St = st.slider(
    "Underlying spot price $S_0$",
    min_value=5.0,
    max_value=500.0,
    value=100.0,
    step=0.01,
)

K = St
st.metric(label="Option strike $K$ = St(starts at-the-money)", value=St)

T_int = st.slider(
    "Maturity $T$ (years)",
    min_value=1,
    max_value=5,
    value=1,
    step=1,
)

T = float(T_int)

r = st.slider(
    "Risk-free rate $r$ (continuous, per year)",
    min_value=0.01,
    max_value=0.10,
    value=0.05,
    step=0.005,
)

sigma = st.slider(
    "Volatility $\\sigma$",
    min_value=0.0,
    max_value=1.0,
    value=0.20,
    step=0.01,
)

st.subheader("Monte Carlo simulation parameters")

n_steps = st.slider(
    "Number of time steps",
    min_value=10,
    max_value=500,
    value=100,
    step=10,
)

n_paths = st.slider(
    "Number of simulated paths",
    min_value=10,
    max_value=1000,
    value=200,
    step=10,
)

# --- Split initial investment using ZCB PV ----------------------------------

# Target amount to be guaranteed at maturity
target_protection_T = investment_amount * (protection_pct/100.0)


# Present value today of that guaranteed amount (ZCB investment)
zcb_pv = zero_coupon_price(target_protection_T, t, T, r)

# Remaining budget for derivatives today
derivatives_pv = investment_amount - zcb_pv

# Show capital protection at t = 0
protection_df = pd.DataFrame(
    {"Amount": [target_protection_T, (investment_amount*(1-(protection_pct/100.0)))]},
    index=["Minimim capital at maturity", "maximum capital loss at at matieity"],
)

# Show allocation at t = 0
allocation_df = pd.DataFrame(
    {"Amount": [zcb_pv, derivatives_pv]},
    index=["Investment in ZCB (PV today)", "Investment in derivatives (PV today)"],
)


# Display results
st.write(f"Valuation time $t$: {t}, maturity $T$: {T:.1f} years")

st.markdown("### Step 2: Split initial investment into ZCB and derivatives")

st.bar_chart(
    protection_df.style.format({"Amount": "{:.2f}"})
)

st.bar_chart(
    allocation_df.style.format({"Amount": "{:.2f}"})
)

st.write(
    f"**Present value invested in ZCB:** {zcb_pv:,.2f} €  \n"
    f"**Present value invested in derivatives:** {derivatives_pv:,.2f} €"
)

st.info(
    "The ZCB is chosen so that, if held to maturity, it pays the desired "
    f"capital protection of {target_protection_T:,.2f} €."
)

st.markdown("---")

# 3. Option prices at base strike
st.subheader("Option prices At The Money")

call_base = BSM_call_value(St, K, t, T, r, sigma)
put_base = BSM_put_value(St, K, t, T, r, sigma)

col1, col2 = st.columns(2)
with col1:
    st.metric("Call price", f"{call_base:,.2f} €")
with col2:
    st.metric("Put price", f"{put_base:,.2f} €")

# 4. Option prices for strikes around K
st.subheader("Call and put prices for strikes around $K$ coloured based on their moneiness")

num_steps = 5
rel_steps = np.arange(num_steps, -num_steps - 1, -1)
strike_multipliers = 1 + 0.1 * rel_steps #generating strike multipliers to set the different strike values fro the table
strike_grid = K * strike_multipliers

call_prices = [BSM_call_value(St, K_i, t, T, r, sigma) for K_i in strike_grid]
put_prices = [BSM_put_value(St, K_i, t, T, r, sigma) for K_i in strike_grid]

grid_df = pd.DataFrame(
    {"Strike": strike_grid, "Call price": call_prices, "Put price": put_prices}
).sort_values("Strike")

st.dataframe(
    grid_df.style.format(
        {"Strike": "{:.2f}", "Call price": "{:.2f}", "Put price": "{:.2f}"}
    ).background_gradient(
        subset=["Call price", "Put price"], cmap="YlGnBu"
    )
)

st.markdown("---")

st.subheader("Step 3: Simulate underlying price paths using GBM")
st.subheader("GBM simulation: paths and terminal distribution")

# Only proceed if we have simulation results

# Use r as drift (risk-neutral) or use another mu if you decide so
time, paths, terminal_prices = simulate_gbm_paths(
    S0=St,      # your current spot slider
    T=T,
    r=r,
    sigma=sigma,
    n_steps=n_steps,
    n_paths=n_paths,
)

st.success("Simulation completed.")
# Create two columns
col1, col2 = st.columns(2)

# --- LEFT COLUMN: price paths ----------------------------------------- #
with col1:
    st.markdown("**Simulated price paths**")

    # Put paths into a DataFrame for Streamlit plotting
    paths_df = pd.DataFrame(paths, index=time)

    # To keep it readable, only plot the first N paths
    max_to_show = min(100, paths_df.shape[1])
    st.line_chart(paths_df.iloc[:, :max_to_show])

# --- RIGHT COLUMN: histogram rotated (prices on y-axis) --------------- #
with col2:
    st.markdown("**Distribution of terminal prices $S_T$**")

    # Compute histogram
    hist_values, bin_edges = np.histogram(terminal_prices, bins=30)

    # Bin centers for plotting
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Create horizontal bar chart with matplotlib
    fig, ax = plt.subplots()
    # barh(y, width)
    ax.barh(bin_centers, hist_values, height=bin_edges[1] - bin_edges[0])

    ax.set_xlabel("Count")         # x-axis = frequency
    ax.set_ylabel("Price $S_T$")   # y-axis = price
    ax.set_title("Histogram of terminal prices")

    st.pyplot(fig)

    # Optional: basic stats
    st.write(
        f"Mean $S_T$: {terminal_prices.mean():.2f}  \n"
        f"Std $S_T$: {terminal_prices.std():.2f}"
    )

log_returns = np.log(terminal_prices / St)

hist_values, bin_edges = np.histogram(log_returns, bins=30)

# Bin centers and width
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
bin_width = bin_edges[1] - bin_edges[0]

fig, ax = plt.subplots()
ax.bar(bin_centers, hist_values, width=bin_width)

ax.set_xlabel("Log-return ln(S_T / S_0)")
ax.set_ylabel("Count")
ax.set_title("Histogram of simulated log-returns")

st.pyplot(fig)



st.write(
    f"Mean log-return: {log_returns.mean():.4f}  \n"
    f"Std of log-returns: {log_returns.std():.4f}"
)

st.subheader("Bond + Call")

st.subheader("Strike ladder with moneyness")

st.subheader("Select a strike")

# Build labels like "K = 90.00 (call 5.23 €)"

Proportion_options = [
    call_prices[i]/derivatives_pv for i in range(len(call_prices))
    ]

strike_options = [
    f"K = {K_i:.2f} (call {c_i:.2f} €) = derivative budget {d_i:2f}"
    for K_i, c_i,d_i in zip(strike_grid, call_prices,Proportion_options)
]

selected_label = st.radio(
    "Choose a strike for the call you buy:",
    options=strike_options,
    
    index=4,
)

# Find which index was selected
selected_index = strike_options.index(selected_label)

# Get the numeric strike and premium
K_selected = float(strike_grid[selected_index])
call_premium = float(call_prices[selected_index])
selected_proportion = float(Proportion_options[selected_index])




st.write(f"**Selected strike K:** {K_selected:.2f}")
st.write(f"**Call premium paid today (per unit):** {call_premium:.2f} €")

st.subheader("Payoff at maturity for the selected call")

# Range of possible underlying prices at maturity
S_T = np.linspace(0.1 * St, 3 * St, 100)  # from 50% of St to 150% of St

# Payoff of 1 long call (net of premium)
payoff = (np.maximum(S_T - K_selected, 0) - call_premium)* selected_proportion

# Put into a DataFrame for plotting
payoff_df = pd.DataFrame(
    {"S_T": S_T, "Net payoff": payoff}
).set_index("S_T")

st.line_chart(payoff_df)



call_payoff_distribution = (np.maximum(terminal_prices - K_selected, 0) - call_premium) * selected_proportion

structured_product_call_distribution = np.log((call_payoff_distribution + target_protection_T)/investment_amount)

col1,col2 = st.columns(2)
with col1:
    hist_values, bin_edges = np.histogram(log_returns, bins=30)

    # Bin centers and width
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_width = bin_edges[1] - bin_edges[0]

    fig, ax = plt.subplots()
    ax.bar(bin_centers, hist_values, width=bin_width)

    ax.set_xlabel("Log-return ln(S_T / S_0)")
    ax.set_ylabel("Count")
    ax.set_title("Histogram of simulated direct investment log-returns")

    st.pyplot(fig)

    st.write(
        f"Mean log-return: {log_returns.mean():.4f}  \n"
        f"Std of log-returns: {log_returns.std():.4f}"
    )
with col2:
    hist_values, bin_edges = np.histogram(structured_product_call_distribution, bins=30)

    # Bin centers and width
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_width = bin_edges[1] - bin_edges[0]

    fig, ax = plt.subplots()
    ax.bar(bin_centers, hist_values, width=bin_width)

    ax.set_xlabel("Structured product return")
    ax.set_ylabel("Count")
    ax.set_title("Histogram of simulated structured product log-returns")

    st.pyplot(fig)

    st.write(
        f"Mean log-return: {structured_product_call_distribution.mean():.4f}  \n"
        f"Std of log-returns: {structured_product_call_distribution.std():.4f}"
    )