# European options are suitable for BSM as they cannot be exercised before the expiration date
# Call options only
# curr_call_price in $, curr_underlying_price in $, strike_price in $, vol in %, rr in %, t in years
# N = standard normal cumulative distributive function
# d1 = [ ln(curr_price / strike_price) + (rr + ((vol^2) / 2)) t ] / (vol * sqrt(t))
# d2 = d1 - (vol * sqrt(t))
# call_value = (curr_price * N(d1)) - (strike_price * (exp ^ (rr * t)) * N(d2))
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.stats import norm
import yfinance as yf
import streamlit as st

# BSM pricing model
def calc(curr_underlying_price, strike_price, vol, rr, t):
    rr = rr / 100
    vol = vol / 100 
    d1 = (math.log(curr_underlying_price / strike_price) + (rr + ((math.pow(vol, 2)) / 2)) * t) / (vol * math.sqrt(t))
    d2 = d1 - (vol * math.sqrt(t))
    N_d1 = 0.5 * (1.0 + math.erf(d1 / math.sqrt(2.0)))
    N_d2 = 0.5 * (1.0 + math.erf(d2 / math.sqrt(2.0)))
    call_value = (curr_underlying_price * N_d1) - (strike_price * (math.exp(-rr * t)) * N_d2)
    return d1, d2, N_d1, N_d2, call_value

# greeks are essential in understanding risk management
def greeks(d1, d2, N_d1, N_d2, curr_underlying_price, strike_price, vol, rr, t):
    # delta = N_d1
    # gamma = N'_d1 / (curr_underlying_price * vol * sqrt(t)) where N'_d1 = (1 / (2root(pi))) * (e ^ ((-d1 * d1) / 2))
    # vega = (curr_underlying_price * N'_d1 * sqrt(t)) / 100
    # theta = -(curr_underlying_price * N'_d1 * vol) / (2 * sqrt(t))) - (rr * strike_price * (e ^ (-rr * t)) * N_d2)
    # rho = strike_price * t * (e ^ (-rr * t)) * N_d2
    rr = rr / 100
    vol = vol / 100
    delta = N_d1
    N_dash_d1 = norm.pdf(d1)
    gamma = N_dash_d1 / (curr_underlying_price * vol * math.sqrt(t))
    vega = (curr_underlying_price * N_dash_d1 * math.sqrt(t)) / 100
    theta = (((-curr_underlying_price * N_dash_d1 * vol) / (2 * math.sqrt(t))) - (rr * strike_price * (math.exp(-rr * t)) * N_d2)) / 365
    rho = (strike_price * t * (math.exp(-rr * t)) * N_d2) / 100
    return delta, gamma, vega, theta, rho

def fetch_dax30_data(ticker='^GDAXI', start_date='2022-01-01', end_date='2023-01-01'):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

def calculate_volatility(data):
    data['Log Return'] = np.log(data['Close'] / data['Close'].shift(1))
    volatility = data['Log Return'].std() * np.sqrt(252)  # Annualize the volatility
    return volatility

# volga is the second order derivative of vega. Plotting volga with respect to volatility helps the trader understand how the vega might react to different volatitlites of the DAX30 market, ultimately affecting the option price.
# A trader may evaluate the implied volatility and compare it with the graph to understand whether the trade is worth taking.
# Of course, this is based on past historical data and is subject to change.
def calc_volga(curr_underlying_price, strike_price, vol, rr, t):
    vol = vol / 100
    rr = rr / 100
    d1 = (np.log(curr_underlying_price / strike_price) + (rr + 0.5 * vol ** 2) * t) / (vol * np.sqrt(t))
    d2 = d1 - vol * np.sqrt(t)
    N_dash_d1 = norm.pdf(d1)
    volga = N_dash_d1 * np.sqrt(t) * (d1 * d2 - 1) / vol
    return volga

def display(curr_underlying_price, strike_price, rr, t):
    dax30_data = fetch_dax30_data()
    historical_vol = calculate_volatility(dax30_data)

    fig, ax = plt.subplots()
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.set_title('Volga vs. Volatility', color='white')
    ax.set_xlabel('Volatility (%)', color='white')
    ax.set_ylabel('Volga', color='white')

    volatilities = np.linspace(1, 100, 100)
    volgas = [calc_volga(curr_underlying_price, strike_price, vol, rr, t) for vol in volatilities]
    scatter = ax.scatter(volatilities, volgas, c=volatilities, cmap='viridis', edgecolors='w')

    def animate(i):
        colors = np.sin(np.linspace(0, 2 * np.pi, len(volatilities)) + i / 10)
        scatter.set_array(colors)
        fig.canvas.draw_idle()

    anim = FuncAnimation(fig, animate, interval=100)
    
    img_path = "volga_vs_volatility.png"
    plt.savefig(img_path)
    plt.close(fig)
    
    return img_path

def simulate_future_prices(curr_underlying_price, historical_volatility, months=12):
    future_prices = []
    price = curr_underlying_price
    for _ in range(months):
        price += np.random.normal(0, historical_volatility)
        future_prices.append(price)
    return future_prices

# A hedging strategy is often used by new and experienced traders alike, a good hedge will minimize risk to both directions of the market.
# The code generates a prediction for the delta each month based on past historical data and rehedges to return to delta neutral.
# A trader can use this table as a rough guidline as to how the option may perform and the size of adjustments that must be made.
# The table is based on past data and so, a trader must perform his own evaluations before executing a trade.
def calculate_pnl_and_hedges(curr_underlying_price, strike_price, vol, rr, t, num_options, historical_volatility, months=12):
    future_prices = simulate_future_prices(curr_underlying_price, historical_volatility, months)
    future_volatilities = np.full(months, vol)
    
    data = []
    hedge_positions = []
    initial_delta = calc(curr_underlying_price, strike_price, vol, rr, t)[0]
    initial_hedge = initial_delta * num_options
    hedge_positions.append(initial_hedge)
    
    total_pnl = 0

    # Adding Month 0
    data.append({
        "Month": 0,
        "Price": curr_underlying_price,
        "Delta": initial_delta,
        "Hedge Position": initial_hedge,
        "PnL": 0
    })

    for i in range(months):
        t_left = (months - i) / 12
        price = future_prices[i]
        vol = future_volatilities[i]
        delta = calc(price, strike_price, vol, rr, t_left)[0]
        hedge = delta * num_options
        hedge_positions.append(hedge)
        pnl = (hedge_positions[-1] - hedge_positions[-2]) * (price - curr_underlying_price)
        total_pnl += pnl
        data.append({
            "Month": i + 1,
            "Price": price,
            "Delta": delta,
            "Hedge Position": hedge,
            "PnL": pnl
        })
        curr_underlying_price = price  

    data.append({
        "Month": "Total",
        "Price": "-",
        "Delta": "-",
        "Hedge Position": "-",
        "PnL": total_pnl
    })

    return pd.DataFrame(data)

def plot_hedging_table(df):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')
    ax.axis('tight')
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
    table.scale(1, 1.5)
    plt.title("Hedging Strategy Table")
    
    img_path = "hedging_table.png"
    plt.savefig(img_path)
    plt.close(fig)
    
    return img_path

# The main function that takes all the user inputs and executes the code.
def main():
    st.title("Option Hedging Strategy")

    curr_call_price = st.text_input("Current Call Price in $", "")
    curr_underlying_price = st.text_input("Current Price of the Underlying $", "")
    strike_price = st.text_input("Strike Price in $", "")
    vol = st.text_input("Volatility in %", "")
    rr = st.text_input("Risk-Free Interest Rate in %", "")
    t = st.text_input("Time to Maturity in Years", "")
    num_options = st.text_input("Number of Options", "")

    if st.button("Calculate"):
        if all([curr_call_price, curr_underlying_price, strike_price, vol, rr, t, num_options]):
            curr_call_price = float(curr_call_price)
            curr_underlying_price = float(curr_underlying_price)
            strike_price = float(strike_price)
            vol = float(vol)
            rr = float(rr)
            t = float(t)
            num_options = float(num_options)

            # Fetch historical data and calculate volatility
            dax30_data = fetch_dax30_data()
            historical_volatility = calculate_volatility(dax30_data)
            
            # Calculate the call value and Greeks
            d1, d2, N_d1, N_d2, call_value = calc(curr_underlying_price, strike_price, vol, rr, t)
            delta, gamma, vega, theta, rho = greeks(d1, d2, N_d1, N_d2, curr_underlying_price, strike_price, vol, rr, t)

            # Display calculated call value and Greeks
            st.write(f"Calculated Call Value: ${call_value:.2f}")
            st.write(f"Delta: {delta:.4f}")
            st.write(f"Gamma: {gamma:.4f}")
            st.write(f"Vega: {vega:.4f}")
            st.write(f"Theta: {theta:.4f}")
            st.write(f"Rho: {rho:.4f}")

            # Calculate hedge and PnL
            df = calculate_pnl_and_hedges(curr_underlying_price, strike_price, vol, rr, t, num_options, historical_volatility)
            
            st.write(df)

            # Plotting table
            img_path = plot_hedging_table(df)
            st.image(img_path)

            # Plotting Volga vs Volatility
            img_path = display(curr_underlying_price, strike_price, rr, t)
            st.image(img_path)

if __name__ == "__main__":
    main()
