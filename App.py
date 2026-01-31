import requests
import pandas as pd
import numpy as np
import time
import datetime
from deap import base, creator, tools, algorithms
import random
from flask import Flask, render_template_string
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import threading
import logging

# --- CONFIGURATION ---
SYMBOL = 'ETHUSDT'
INTERVAL = '1h'
YEARS = 3
PORT = 8080
POPULATION_SIZE = 50
GENERATIONS = 10
hof = tools.HallOfFame(1)

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# --- 1. DATA FETCHING ---
def fetch_binance_data(symbol, interval, years):
    base_url = "https://api.binance.com/api/v3/klines"
    end_time = int(time.time() * 1000)
    start_time = int((datetime.datetime.now() - datetime.timedelta(days=years*365)).timestamp() * 1000)
    
    all_data = []
    current_start = start_time
    
    logger.info(f"Fetching {years} years of {interval} data for {symbol}...")
    
    while current_start < end_time:
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': current_start,
            'limit': 1000
        }
        response = requests.get(base_url, params=params)
        data = response.json()
        
        if not data:
            break
            
        all_data.extend(data)
        current_start = data[-1][0] + 1
        time.sleep(0.1) # Rate limit handling
        
    df = pd.DataFrame(all_data, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume', 
        'close_time', 'qav', 'num_trades', 'taker_base_vol', 'taker_quote_vol', 'ignore'
    ])
    
    df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    
    df.set_index('timestamp', inplace=True)
    logger.info(f"Fetched {len(df)} candles.")
    return df[['open', 'high', 'low', 'close']]

# --- 2. QUARTER SPLITTING & UTILS ---
def split_into_cycles(df):
    """
    Splits data into 4-month cycles.
    Structure: 
    - Month 1-2: Train (GA)
    - Month 3: Gap (Ignored per prompt implication or used as validation, but prompt skips it)
    - Month 4: Test
    """
    # Group by year and month
    groups = [g for n, g in df.groupby(pd.Grouper(freq='M'))]
    
    cycles = []
    # We need chunks of 4 months
    for i in range(0, len(groups) - 3, 4):
        cycle_data = {
            'train': pd.concat([groups[i], groups[i+1]]),
            'gap': groups[i+2],
            'test': groups[i+3],
            'id': f"Cycle_{i//4 + 1}_{groups[i].index[0].strftime('%Y-%b')}"
        }
        cycles.append(cycle_data)
    return cycles

# --- 3. STRATEGY & BACKTESTING ---
def backtest(df, params):
    """
    Params: [buy_reversal_pct, sell_reversal_pct, stop_loss_pct, take_profit_pct]
    Strategy: 
    - Calculate a rolling mean (simple anchor).
    - Buy if Price < Mean * (1 - buy_reversal_pct)
    - Sell (Short) if Price > Mean * (1 + sell_reversal_pct)
    - Exit on SL or TP.
    """
    buy_rev, sell_rev, sl, tp = params
    
    # Vectorized signal generation is hard with complex SL/TP path dependency, 
    # employing fast iteration with numba is ideal, but using standard loop for compatibility.
    
    closes = df['close'].values
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    times = df.index
    
    # Simple Mean Reversion Reference: 24h SMA
    sma = df['close'].rolling(window=24).mean().fillna(method='bfill').values
    
    position = 0 # 1: Long, -1: Short, 0: Flat
    entry_price = 0.0
    equity = [1000.0] # Start with $1000
    trades = []
    
    for i in range(len(closes)):
        current_price = closes[i]
        current_sma = sma[i]
        curr_equity = equity[-1]
        
        # Check Exits first
        if position == 1:
            # Long Exit
            if lows[i] <= entry_price * (1 - sl): # Stop Loss
                curr_equity *= (1 - sl)
                position = 0
                trades.append({'time': times[i], 'type': 'exit_sl_long', 'price': entry_price * (1 - sl), 'pnl': -sl})
            elif highs[i] >= entry_price * (1 + tp): # Take Profit
                curr_equity *= (1 + tp)
                position = 0
                trades.append({'time': times[i], 'type': 'exit_tp_long', 'price': entry_price * (1 + tp), 'pnl': tp})
                
        elif position == -1:
            # Short Exit
            if highs[i] >= entry_price * (1 + sl): # Stop Loss
                curr_equity *= (1 - sl)
                position = 0
                trades.append({'time': times[i], 'type': 'exit_sl_short', 'price': entry_price * (1 + sl), 'pnl': -sl})
            elif lows[i] <= entry_price * (1 - tp): # Take Profit
                curr_equity *= (1 + tp)
                position = 0
                trades.append({'time': times[i], 'type': 'exit_tp_short', 'price': entry_price * (1 - tp), 'pnl': tp})

        # Check Entries (only if flat)
        if position == 0:
            # Buy Limit
            target_buy = current_sma * (1 - buy_rev)
            if lows[i] <= target_buy:
                position = 1
                entry_price = target_buy
                trades.append({'time': times[i], 'type': 'long', 'price': entry_price})
            
            # Sell Limit
            target_sell = current_sma * (1 + sell_rev)
            if highs[i] >= target_sell:
                position = -1
                entry_price = target_sell
                trades.append({'time': times[i], 'type': 'short', 'price': entry_price})

        equity.append(curr_equity)
        
    returns = pd.Series(equity).pct_change().fillna(0)
    sharpe = np.sqrt(len(equity)) * (returns.mean() / returns.std()) if returns.std() != 0 else 0
    return sharpe, equity, trades

# --- 4. GENETIC ALGORITHM ---
# Genes: [Buy_Rev%, Sell_Rev%, SL%, TP%]
# Bounds: Rev (0.001 - 0.05), SL (0.001 - 0.05), TP (0.005 - 0.10)

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_rev", random.uniform, 0.001, 0.05)
toolbox.register("attr_sl", random.uniform, 0.005, 0.03) # Tighter SL
toolbox.register("attr_tp", random.uniform, 0.01, 0.10)  # Wider TP

toolbox.register("individual", tools.initCycle, creator.Individual, 
                 (toolbox.attr_rev, toolbox.attr_rev, toolbox.attr_sl, toolbox.attr_tp), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evaluate(individual, data):
    # Only return fitness (Sharpe)
    sharpe, _, _ = backtest(data, individual)
    return (sharpe,)

toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.01, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

def run_ga(train_data):
    # Bind the specific data to the evaluation function
    toolbox.register("evaluate", evaluate, data=train_data)
    
    pop = toolbox.population(n=POPULATION_SIZE)
    
    # Add bounds handling to mutation
    def checkBounds(min_v, max_v):
        def decorator(func):
            def wrapper(*args, **kargs):
                offspring = func(*args, **kargs)
                for child in offspring:
                    for i in range(len(child)):
                        if child[i] < min_v: child[i] = min_v
                        if child[i] > max_v: child[i] = max_v
                return offspring
            return wrapper
        return decorator

    toolbox.decorate("mutate", checkBounds(0.001, 0.15))
    
    final_pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, 
                                         ngen=GENERATIONS, stats=None, 
                                         halloffame=hof, verbose=False)
    return hof[0]

# --- 5. MAIN EXECUTION LOOP ---
results_store = []

def process_data():
    global results_store
    df = fetch_binance_data(SYMBOL, INTERVAL, YEARS)
    cycles = split_into_cycles(df)
    
    cumulative_equity = 1000.0
    
    logger.info(f"Identified {len(cycles)} backtesting cycles.")
    
    for cycle in cycles:
        train_df = cycle['train']
        test_df = cycle['test']
        
        logger.info(f"Optimizing {cycle['id']} (Train: {len(train_df)} hrs)...")
        
        # Optimize on Months 1 & 2
        best_ind = run_ga(train_df)
        
        # Test on Month 4
        logger.info(f"Testing {cycle['id']} (Test: {len(test_df)} hrs)... params: {[round(x,4) for x in best_ind]}")
        sharpe, equity_curve, trades = backtest(test_df, best_ind)
        
        # Calculate real PnL for this segment
        segment_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0]
        cumulative_equity *= (1 + segment_return)
        
        results_store.append({
            'cycle': cycle['id'],
            'params': best_ind,
            'sharpe': sharpe,
            'segment_pnl_pct': segment_return * 100,
            'cumulative_equity': cumulative_equity,
            'test_data': test_df,
            'trades': trades
        })

# --- 6. FLASK SERVER & VISUALIZATION ---
app = Flask(__name__)

@app.route('/')
def dashboard():
    if not results_store:
        return "<h1>Processing Data... Please refresh in a minute.</h1>"
    
    # Create Plotly Graph
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.03, subplot_titles=('Price & Entries', 'Cumulative Equity'),
                        row_width=[0.3, 0.7])

    # Concatenate test data for visualization
    combined_test_df = pd.DataFrame()
    
    # Plotting loop
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown'] * 10
    
    current_eq_base = 1000
    
    for idx, res in enumerate(results_store):
        df = res['test_data']
        trades = res['trades']
        
        # Candlestick
        fig.add_trace(go.Candlestick(x=df.index,
                        open=df['open'], high=df['high'],
                        low=df['low'], close=df['close'],
                        name=f"{res['cycle']} Price"), row=1, col=1)
        
        # Entries/Exits Markers
        for t in trades:
            color = 'Green' if 'long' in t['type'] and 'exit' not in t['type'] else \
                    'Red' if 'short' in t['type'] and 'exit' not in t['type'] else \
                    'Black' # Exits
            
            symbol = 'triangle-up' if 'long' in t['type'] and 'exit' not in t['type'] else \
                     'triangle-down' if 'short' in t['type'] and 'exit' not in t['type'] else \
                     'x'
                     
            fig.add_trace(go.Scatter(x=[t['time']], y=[t['price']], mode='markers',
                                     marker=dict(symbol=symbol, color=color, size=10),
                                     showlegend=False,
                                     name=t['type']), row=1, col=1)

        # Equity Curve (reconstructed roughly for visualization)
        # We just plot the end point of each cycle to show growth
        fig.add_trace(go.Scatter(x=[df.index[-1]], y=[res['cumulative_equity']], 
                                 mode='markers+lines', marker=dict(color='blue'),
                                 name='Equity Checkpoint'), row=2, col=1)

    fig.update_layout(height=800, title_text="ETH 3-Year Quarterly Walk-Forward (Train M1-2, Test M4)", template="plotly_dark")
    
    graph_html = fig.to_html(full_html=False)
    
    # Stats Table
    stats_html = """
    <table border="1" style="border-collapse: collapse; width: 100%; color: white; font-family: sans-serif;">
        <tr style="background-color: #333;">
            <th>Cycle</th><th>Buy Rev %</th><th>Sell Rev %</th><th>SL %</th><th>TP %</th><th>Test Sharpe</th><th>PnL %</th><th>Cum Equity</th>
        </tr>
    """
    for res in results_store:
        p = res['params']
        stats_html += f"""
        <tr>
            <td>{res['cycle']}</td>
            <td>{p[0]*100:.2f}%</td>
            <td>{p[1]*100:.2f}%</td>
            <td>{p[2]*100:.2f}%</td>
            <td>{p[3]*100:.2f}%</td>
            <td>{res['sharpe']:.4f}</td>
            <td style="color: {'lime' if res['segment_pnl_pct'] > 0 else 'red'}">{res['segment_pnl_pct']:.2f}%</td>
            <td>${res['cumulative_equity']:.2f}</td>
        </tr>
        """
    stats_html += "</table>"

    html = f"""
    <html>
        <head><title>Bot Results</title></head>
        <body style="background-color: #111; color: #ddd;">
            <h1 style="text-align:center">Genetic Algorithm Strategy Backtest</h1>
            {graph_html}
            <div style="padding: 20px;">
                {stats_html}
            </div>
        </body>
    </html>
    """
    return render_template_string(html)

def run_server():
    app.run(host='0.0.0.0', port=PORT, debug=False, use_reloader=False)

if __name__ == "__main__":
    # Start data processing in background
    t = threading.Thread(target=process_data)
    t.start()
    
    # Start Web Server
    print(f"Server starting on http://localhost:{PORT}")
    run_server()
