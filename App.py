import requests
import pandas as pd
import numpy as np
import time
import datetime
from deap import base, creator, tools, algorithms
import random
from flask import Flask, render_template_string, Response, stream_with_context
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import threading
import logging
import queue
import json

# --- CONFIGURATION ---
SYMBOL = 'ETHUSDT'
INTERVAL = '1h'
YEARS = 3
PORT = 8080
POPULATION_SIZE = 50
GENERATIONS = 10

# --- LOGGING & SSE SETUP ---
# Thread-safe queue for log messages
log_queue = queue.Queue()

class QueueHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            log_queue.put(msg)
        except Exception:
            self.handleError(record)

# Configure Logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# Handler 1: Standard Output
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# Handler 2: Queue for Web
queue_handler = QueueHandler()
queue_handler.setFormatter(formatter)
logger.addHandler(queue_handler)

hof = tools.HallOfFame(1)

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
        try:
            response = requests.get(base_url, params=params)
            data = response.json()
            
            if not data or not isinstance(data, list):
                break
                
            all_data.extend(data)
            current_start = data[-1][0] + 1
            time.sleep(0.1) 
        except Exception as e:
            logger.error(f"Fetch error: {e}")
            break
        
    if not all_data:
        logger.error("No data fetched.")
        return pd.DataFrame()

    df = pd.DataFrame(all_data, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume', 
        'close_time', 'qav', 'num_trades', 'taker_base_vol', 'taker_quote_vol', 'ignore'
    ])
    
    df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    
    df.set_index('timestamp', inplace=True)
    logger.info(f"Fetched {len(df)} total candles.")
    return df[['open', 'high', 'low', 'close']]

# --- 2. QUARTER SPLITTING & UTILS ---
def split_into_cycles(df):
    groups = [g for n, g in df.groupby(pd.Grouper(freq='M'))]
    cycles = []
    # Train: M1+M2, Gap: M3, Test: M4
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
    buy_rev, sell_rev, sl, tp = params
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    times = df.index
    
    # 24h SMA Reference
    sma = df['close'].rolling(window=24).mean().fillna(method='bfill').values
    
    position = 0 
    entry_price = 0.0
    equity = [1000.0]
    trades = []
    
    for i in range(len(closes)):
        current_sma = sma[i]
        curr_equity = equity[-1]
        
        if position == 1:
            if lows[i] <= entry_price * (1 - sl):
                curr_equity *= (1 - sl)
                position = 0
                trades.append({'time': times[i], 'type': 'exit_sl_long', 'price': entry_price * (1 - sl), 'pnl': -sl})
            elif highs[i] >= entry_price * (1 + tp):
                curr_equity *= (1 + tp)
                position = 0
                trades.append({'time': times[i], 'type': 'exit_tp_long', 'price': entry_price * (1 + tp), 'pnl': tp})
                
        elif position == -1:
            if highs[i] >= entry_price * (1 + sl):
                curr_equity *= (1 - sl)
                position = 0
                trades.append({'time': times[i], 'type': 'exit_sl_short', 'price': entry_price * (1 + sl), 'pnl': -sl})
            elif lows[i] <= entry_price * (1 - tp):
                curr_equity *= (1 + tp)
                position = 0
                trades.append({'time': times[i], 'type': 'exit_tp_short', 'price': entry_price * (1 - tp), 'pnl': tp})

        if position == 0:
            target_buy = current_sma * (1 - buy_rev)
            if lows[i] <= target_buy:
                position = 1
                entry_price = target_buy
                trades.append({'time': times[i], 'type': 'long', 'price': entry_price})
            
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
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_rev", random.uniform, 0.001, 0.05)
toolbox.register("attr_sl", random.uniform, 0.005, 0.03)
toolbox.register("attr_tp", random.uniform, 0.01, 0.10)

toolbox.register("individual", tools.initCycle, creator.Individual, 
                 (toolbox.attr_rev, toolbox.attr_rev, toolbox.attr_sl, toolbox.attr_tp), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evaluate(individual, data):
    sharpe, _, _ = backtest(data, individual)
    return (sharpe,)

toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.01, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

def run_ga(train_data):
    toolbox.register("evaluate", evaluate, data=train_data)
    pop = toolbox.population(n=POPULATION_SIZE)
    
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
    
    # Custom Algorithm to allow logging per generation
    # Replaces algorithms.eaSimple to inject logs
    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, 
                                       ngen=GENERATIONS, verbose=False)
    
    return hof[0]

# --- 5. EXECUTION & STATE ---
results_store = []
processing_complete = False

def process_data():
    global results_store, processing_complete
    logger.info("Background thread started.")
    
    df = fetch_binance_data(SYMBOL, INTERVAL, YEARS)
    if df.empty:
        logger.error("Dataframe empty. Aborting.")
        return

    cycles = split_into_cycles(df)
    cumulative_equity = 1000.0
    
    logger.info(f"Identified {len(cycles)} backtesting cycles.")
    
    for idx, cycle in enumerate(cycles):
        train_df = cycle['train']
        test_df = cycle['test']
        
        logger.info(f"--- Processing {cycle['id']} ({idx+1}/{len(cycles)}) ---")
        logger.info(f"Training on {len(train_df)} candles (Months 1-2)...")
        
        best_ind = run_ga(train_df)
        
        logger.info(f"Best Params Found: BuyRev={best_ind[0]:.4f}, SellRev={best_ind[1]:.4f}, SL={best_ind[2]:.4f}, TP={best_ind[3]:.4f}")
        logger.info(f"Testing on {len(test_df)} candles (Month 4)...")
        
        sharpe, equity_curve, trades = backtest(test_df, best_ind)
        segment_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0]
        cumulative_equity *= (1 + segment_return)
        
        logger.info(f"Cycle Result: Sharpe={sharpe:.2f}, PnL={segment_return*100:.2f}%, Cumulative Eq=${cumulative_equity:.2f}")
        
        results_store.append({
            'cycle': cycle['id'],
            'params': best_ind,
            'sharpe': sharpe,
            'segment_pnl_pct': segment_return * 100,
            'cumulative_equity': cumulative_equity,
            'test_data': test_df,
            'trades': trades
        })
        
    logger.info("Processing Complete. Refresh page to see full charts.")
    processing_complete = True

# --- 6. FLASK & VISUALIZATION ---
app = Flask(__name__)

@app.route('/stream')
def stream_logs():
    def generate():
        while True:
            try:
                # Non-blocking get
                message = log_queue.get(timeout=1.0)
                yield f"data: {message}\n\n"
            except queue.Empty:
                # Send keep-alive to prevent timeout
                yield f": keep-alive\n\n"
    return Response(stream_with_context(generate()), mimetype='text/event-stream')

@app.route('/')
def dashboard():
    graph_html = ""
    stats_html = ""
    
    if results_store:
        # Generate Plotly Graphs
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.03, subplot_titles=('Price & Entries', 'Cumulative Equity'),
                            row_width=[0.3, 0.7])

        for idx, res in enumerate(results_store):
            df = res['test_data']
            trades = res['trades']
            
            # Candles
            fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'],
                            low=df['low'], close=df['close'], name=f"{res['cycle']}"), row=1, col=1)
            
            # Trades
            for t in trades:
                color = 'Green' if 'long' in t['type'] and 'exit' not in t['type'] else \
                        'Red' if 'short' in t['type'] and 'exit' not in t['type'] else 'Black'
                symbol = 'triangle-up' if 'long' in t['type'] else 'triangle-down' if 'short' in t['type'] else 'x'
                         
                fig.add_trace(go.Scatter(x=[t['time']], y=[t['price']], mode='markers',
                                         marker=dict(symbol=symbol, color=color, size=8),
                                         showlegend=False, name=t['type']), row=1, col=1)

            # Equity
            fig.add_trace(go.Scatter(x=[df.index[-1]], y=[res['cumulative_equity']], 
                                     mode='markers+lines', marker=dict(color='cyan'),
                                     name='Equity Checkpoint'), row=2, col=1)

        fig.update_layout(height=700, template="plotly_dark", title="Backtest Results")
        graph_html = fig.to_html(full_html=False)
        
        # Stats Table
        rows = ""
        for res in results_store:
            p = res['params']
            rows += f"""
            <tr>
                <td>{res['cycle']}</td>
                <td>{p[0]*100:.2f}% / {p[1]*100:.2f}%</td>
                <td>{p[2]*100:.2f}% / {p[3]*100:.2f}%</td>
                <td>{res['sharpe']:.2f}</td>
                <td style="color:{'lime' if res['segment_pnl_pct']>0 else 'red'}">{res['segment_pnl_pct']:.2f}%</td>
                <td>${res['cumulative_equity']:.0f}</td>
            </tr>"""
        
        stats_html = f"""
        <table class="table">
            <tr><th>Cycle</th><th>Rev (B/S)</th><th>SL/TP</th><th>Sharpe</th><th>PnL</th><th>Equity</th></tr>
            {rows}
        </table>
        """

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Algorithmic Console</title>
        <style>
            body {{ background-color: #121212; color: #e0e0e0; font-family: 'Courier New', monospace; margin: 0; padding: 20px; }}
            .container {{ max_width: 1400px; margin: auto; display: grid; grid-template-columns: 1fr 350px; gap: 20px; }}
            .main-content {{ grid-column: 1; }}
            .sidebar {{ grid-column: 2; }}
            
            #console {{ 
                background: #000; border: 1px solid #333; height: 800px; 
                overflow-y: scroll; padding: 10px; font-size: 12px; line-height: 1.4;
                color: #0f0; white-space: pre-wrap;
            }}
            .table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
            .table th, .table td {{ border: 1px solid #333; padding: 5px; text-align: left; }}
            .table th {{ background: #222; }}
            h1, h2 {{ color: #fff; border-bottom: 1px solid #333; padding-bottom: 5px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="main-content">
                <h1>Results Dashboard</h1>
                {graph_html}
                <h2>Detailed Stats</h2>
                {stats_html}
            </div>
            <div class="sidebar">
                <h2>System Log (Live)</h2>
                <div id="console">Initializing connection...<br></div>
            </div>
        </div>

        <script>
            const consoleDiv = document.getElementById("console");
            const eventSource = new EventSource("/stream");
            
            eventSource.onmessage = function(event) {{
                const newElement = document.createElement("div");
                newElement.textContent = event.data;
                consoleDiv.appendChild(newElement);
                consoleDiv.scrollTop = consoleDiv.scrollHeight; // Auto-scroll
            }};
            
            eventSource.onerror = function() {{
                console.log("Stream connection lost or ended.");
                // Optional: eventSource.close();
            }};
        </script>
    </body>
    </html>
    """
    return render_template_string(html)

if __name__ == "__main__":
    # Start Processing Thread
    t = threading.Thread(target=process_data)
    t.daemon = True
    t.start()
    
    print(f"Server running on port {PORT}")
    app.run(host='0.0.0.0', port=PORT, threaded=True, debug=False, use_reloader=False)
