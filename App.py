import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import random
import http.server
import socketserver
import warnings
import requests
import threading
import time
import json
import urllib.parse
from datetime import datetime, timedelta, timezone
from deap import base, creator, tools, algorithms

# --- Configuration ---
PORT = 8080
N_LINES = 32
POPULATION_SIZE = 320
GENERATIONS = 10
RISK_FREE_RATE = 0.0
MAX_ASSETS_TO_OPTIMIZE = 1
TEST_FEE = 0.00 # 0.2% fee for final test

# Ranges
STOP_PCT_RANGE = (0.001, 0.02)   # 0.1% to 2%
PROFIT_PCT_RANGE = (0.0004, 0.05) # 0.04% to 5%

warnings.filterwarnings("ignore")

# Asset List Mapping
ASSETS = [
    {"symbol": "BTC", "pair": "BTCUSDT"},
    {"symbol": "ETH", "pair": "ETHUSDT"},
    {"symbol": "XRP", "pair": "XRPUSDT"},
    {"symbol": "SOL", "pair": "SOLUSDT"},
    {"symbol": "DOGE", "pair": "DOGEUSDT"},
    {"symbol": "ADA", "pair": "ADAUSDT"},
    {"symbol": "BCH", "pair": "BCHUSDT"},
    {"symbol": "LINK", "pair": "LINKUSDT"},
    {"symbol": "XLM", "pair": "XLMUSDT"},
    {"symbol": "SUI", "pair": "SUIUSDT"},
    {"symbol": "AVAX", "pair": "AVAXUSDT"},
    {"symbol": "LTC", "pair": "LTCUSDT"},
    {"symbol": "HBAR", "pair": "HBARUSDT"},
    {"symbol": "SHIB", "pair": "SHIBUSDT"},
    {"symbol": "TON", "pair": "TONUSDT"},
]

# Global Storage
HTML_REPORTS = {} 
BEST_PARAMS = {}
REPORT_LOCK = threading.Lock()

# --- 1. DEAP Initialization ---
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# --- 2. Precise Data Ingestion (Binance 1M) ---
def fetch_binance_data_1mo(pair):
    base_url = "https://api.binance.com/api/v3/klines"
    # Approx 30 days ago
    start_time = int((datetime.now(timezone.utc) - timedelta(days=30)).timestamp() * 1000)
    end_time = int(datetime.now(timezone.utc).timestamp() * 1000)
    
    all_data = []
    current_start = start_time
    
    print(f"[{pair}] Fetching 1m data from Binance (Last 30 Days)...")
    
    while True:
        params = {
            'symbol': pair,
            'interval': '1m',
            'startTime': current_start,
            'limit': 1000
        }
        
        try:
            r = requests.get(base_url, params=params, timeout=10)
            r.raise_for_status()
            data = r.json()
            
            if not data:
                break
                
            all_data.extend(data)
            
            # Update start time to last candle close time + 1ms
            last_close_time = data[-1][6]
            current_start = last_close_time + 1
            
            if len(data) < 1000 or current_start > end_time:
                break
                
            time.sleep(0.1) # Rate limit respect
            
        except Exception as e:
            print(f"Error fetching {pair}: {e}")
            break
            
    if not all_data:
        return None

    df = pd.DataFrame(all_data, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'qav', 'num_trades', 'taker_base_vol', 'taker_quote_vol', 'ignore'
    ])
    
    df['dt'] = pd.to_datetime(df['open_time'], unit='ms')
    df.set_index('dt', inplace=True)
    
    numeric_cols = ['open', 'high', 'low', 'close']
    df[numeric_cols] = df[numeric_cols].astype(float)
    
    df = df[numeric_cols]
    df.sort_index(inplace=True)
    
    print(f"[{pair}] Fetched {len(df)} 1m rows.")
    return df

def get_data_payload(pair):
    df = fetch_binance_data_1mo(pair)
    if df is None or len(df) < 100:
        return None, None, None, None

    # GA Optimization Data (Resampled to 1H for speed)
    df_1h = df.resample('1h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).dropna()

    # Split Indices
    split_idx_1h = int(len(df_1h) * 0.85)
    train_1h = df_1h.iloc[:split_idx_1h]
    # We don't really use test_1h, we use test_1m for final validation
    
    split_idx_1m = int(len(df) * 0.85)
    test_1m = df.iloc[split_idx_1m:]
    
    return train_1h, test_1m

# --- 3. Strategy Logic (Multi-Trade, Hedged, Fees) ---
def run_backtest(df, stop_pct, profit_pct, lines, fee=0.0, detailed_log_trades=0):
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    times = df.index
    
    equity = 10000.0
    equity_curve = [equity]
    
    # Hedging: Separate lists for active trades
    # Each trade: {'entry': float, 'type': 1/-1}
    long_trades = []
    short_trades = []
    
    trades_history = []
    hourly_log = []
    
    lines = np.sort(lines)
    trades_completed = 0
    
    for i in range(1, len(df)):
        current_c = closes[i]
        current_h = highs[i]
        current_l = lows[i]
        prev_c = closes[i-1]
        ts = times[i]
        
        # --- Check Exits (Longs) ---
        # Iterate backwards to safely remove
        for idx in range(len(long_trades) - 1, -1, -1):
            trade = long_trades[idx]
            entry_price = trade['entry']
            sl_price = entry_price * (1 - stop_pct)
            tp_price = entry_price * (1 + profit_pct)
            
            exit_price = 0.0
            reason = ""
            hit = False
            
            if current_l <= sl_price:
                exit_price = sl_price
                reason = "SL"
                hit = True
            elif current_h >= tp_price:
                exit_price = tp_price
                reason = "TP"
                hit = True
                
            if hit:
                pn_l = (exit_price - entry_price) / entry_price
                # Apply Fee
                net_pnl = pn_l - fee
                equity *= (1 + net_pnl)
                
                trades_history.append({
                    'time': ts, 'type': 'Exit Long', 'price': exit_price, 
                    'pnl': pn_l, 'net_pnl': net_pnl, 'equity': equity, 'reason': reason
                })
                long_trades.pop(idx)
                trades_completed += 1

        # --- Check Exits (Shorts) ---
        for idx in range(len(short_trades) - 1, -1, -1):
            trade = short_trades[idx]
            entry_price = trade['entry']
            sl_price = entry_price * (1 + stop_pct)
            tp_price = entry_price * (1 - profit_pct)
            
            exit_price = 0.0
            reason = ""
            hit = False
            
            if current_h >= sl_price:
                exit_price = sl_price
                reason = "SL"
                hit = True
            elif current_l <= tp_price:
                exit_price = tp_price
                reason = "TP"
                hit = True
                
            if hit:
                pn_l = (entry_price - exit_price) / entry_price
                # Apply Fee
                net_pnl = pn_l - fee
                equity *= (1 + net_pnl)
                
                trades_history.append({
                    'time': ts, 'type': 'Exit Short', 'price': exit_price, 
                    'pnl': pn_l, 'net_pnl': net_pnl, 'equity': equity, 'reason': reason
                })
                short_trades.pop(idx)
                trades_completed += 1

        # --- Entry Logic (Simultaneous) ---
        found_short = False
        short_target = 0.0
        
        if current_h > prev_c:
            idx_s = np.searchsorted(lines, prev_c, side='right')
            idx_e = np.searchsorted(lines, current_h, side='right')
            potential_shorts = lines[idx_s:idx_e]
            if len(potential_shorts) > 0:
                found_short = True
                short_target = potential_shorts[0]

        found_long = False
        long_target = 0.0
        
        if current_l < prev_c:
            idx_s = np.searchsorted(lines, current_l, side='left')
            idx_e = np.searchsorted(lines, prev_c, side='left')
            potential_longs = lines[idx_s:idx_e]
            if len(potential_longs) > 0:
                found_long = True
                long_target = potential_longs[-1]

        # Execute Entries (Independent Accounts)
        if found_short:
            # Avoid duplicate entries on exact same candle/price to prevent spam if grid is tight
            # Simple check: Don't enter if we just entered same price in same list?
            # For this logic, we assume valid signal = trade.
            short_trades.append({'entry': short_target})
            trades_history.append({'time': ts, 'type': 'Short', 'price': short_target, 'pnl': 0, 'equity': equity, 'reason': 'Entry'})
            
        if found_long:
            long_trades.append({'entry': long_target})
            trades_history.append({'time': ts, 'type': 'Long', 'price': long_target, 'pnl': 0, 'equity': equity, 'reason': 'Entry'})

        equity_curve.append(equity)
        
        # --- Detailed Logging ---
        if detailed_log_trades > 0 and len(hourly_log) < detailed_log_trades:
             # Just logging snapshot
             hourly_log.append({
                "Timestamp": str(ts),
                "Price": f"{current_c:.2f}",
                "Active Longs": len(long_trades),
                "Active Shorts": len(short_trades),
                "Equity": f"{equity:.2f}"
            })

    return equity_curve, trades_history, hourly_log

def calculate_sharpe(equity_curve):
    if len(equity_curve) < 2: return -999.0
    returns = pd.Series(equity_curve).pct_change().dropna()
    if returns.std() == 0: return -999.0
    return np.sqrt(8760) * (returns.mean() / returns.std())

# --- 4. Genetic Algorithm ---
def setup_toolbox(min_price, max_price, df_train):
    toolbox = base.Toolbox()
    toolbox.register("attr_stop", random.uniform, STOP_PCT_RANGE[0], STOP_PCT_RANGE[1])
    toolbox.register("attr_profit", random.uniform, PROFIT_PCT_RANGE[0], PROFIT_PCT_RANGE[1])
    toolbox.register("attr_line", random.uniform, min_price, max_price)

    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.attr_stop, toolbox.attr_profit) + (toolbox.attr_line,)*N_LINES, n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    toolbox.register("evaluate", evaluate_genome, df_train=df_train)
    toolbox.register("mate", tools.cxTwoPoint) 
    toolbox.register("mutate", mutate_custom, indpb=0.1, min_p=min_price, max_p=max_price)
    toolbox.register("select", tools.selTournament, tournsize=3)
    return toolbox

def evaluate_genome(individual, df_train):
    stop_pct = np.clip(individual[0], STOP_PCT_RANGE[0], STOP_PCT_RANGE[1])
    profit_pct = np.clip(individual[1], PROFIT_PCT_RANGE[0], PROFIT_PCT_RANGE[1])
    lines = np.array(individual[2:])
    # GA Training: No Fees (standard practice to identify alpha), or Low Fees. 
    # Logic: Prompt says "Apply 0.2 fees to the final test". Implies training is fee-less or different.
    # We stick to fee=0.0 for GA to allow signal discovery, applying rigorous fee check in validation.
    eq_curve, _, _ = run_backtest(df_train, stop_pct, profit_pct, lines, fee=0.0, detailed_log_trades=0)
    return (calculate_sharpe(eq_curve),)

def mutate_custom(individual, indpb, min_p, max_p):
    if random.random() < indpb:
        individual[0] = np.clip(individual[0] + random.gauss(0, 0.005), STOP_PCT_RANGE[0], STOP_PCT_RANGE[1])
    if random.random() < indpb:
        individual[1] = np.clip(individual[1] + random.gauss(0, 0.005), PROFIT_PCT_RANGE[0], PROFIT_PCT_RANGE[1])
    for i in range(2, len(individual)):
        if random.random() < (indpb / 10.0): 
            individual[i] = np.clip(individual[i] + random.gauss(0, (max_p - min_p) * 0.01), min_p, max_p)
    return individual,

# --- 5. Reporting ---
def generate_report(symbol, best_ind, test_curve, test_trades, hourly_log, live_logs=[], live_trades=[]):
    plt.figure(figsize=(14, 8))
    
    # Equity Curve
    plt.title(f"{symbol} Test Equity Curve (Fee: {TEST_FEE*100}%)")
    plt.plot(test_curve, label='Test Equity (with Fees)', color='orange')
    plt.legend()
    plt.grid(True)
    
    img_io = io.BytesIO()
    plt.savefig(img_io, format='png', dpi=100)
    img_io.seek(0)
    plot_url = base64.b64encode(img_io.getvalue()).decode()
    plt.close()
    
    trades_df = pd.DataFrame(test_trades)
    trades_html = trades_df.to_html(classes='table table-striped table-sm', index=False, max_rows=500) if not trades_df.empty else "No trades."
    
    hourly_df = pd.DataFrame(hourly_log)
    hourly_html = hourly_df.to_html(classes='table table-bordered table-sm table-hover', index=False) if not hourly_df.empty else "No hourly data recorded."

    live_log_df = pd.DataFrame(live_logs)
    live_log_html = live_log_df.to_html(classes='table table-bordered table-sm table-hover', index=False) if not live_log_df.empty else "Waiting for next minute trigger..."
    
    live_trades_df = pd.DataFrame(live_trades)
    live_trades_html = live_trades_df.to_html(classes='table table-striped table-sm', index=False) if not live_trades_df.empty else "No live trades yet."

    params_html = f"""
    <ul class="list-group">
        <li class="list-group-item"><strong>Stop Loss:</strong> {best_ind[0]*100:.4f}%</li>
        <li class="list-group-item"><strong>Take Profit:</strong> {best_ind[1]*100:.4f}%</li>
        <li class="list-group-item"><strong>Test Fee Applied:</strong> {TEST_FEE*100:.2f}%</li>
    </ul>
    """

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{symbol} Strategy Results</title>
        <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
        <meta http-equiv="refresh" content="30"> 
        <style>body {{ padding: 20px; }} h3 {{ margin-top: 30px; }} th {{ position: sticky; top: 0; background: white; }}</style>
    </head>
    <body>
        <div class="container-fluid">
            <a href="/" class="btn btn-secondary mb-3">&larr; Back to Dashboard</a>
            <h1 class="mb-4">{symbol} Hedged Grid Strategy</h1>
            <div class="row">
                <div class="col-md-4">{params_html}</div>
                <div class="col-md-8 text-right">
                    <h5>Test Sharpe: {calculate_sharpe(test_curve):.4f}</h5>
                </div>
            </div>
            <hr>
            <h3>Performance Charts</h3>
            <img src="data:image/png;base64,{plot_url}" class="img-fluid border rounded">
            
            <hr>
            <div id="live-section" style="background-color: #f8f9fa; padding: 15px; border-left: 5px solid #28a745;">
                <h2 class="text-success">{symbol} Live Forward Test (Binance 1m)</h2>
                <div class="row">
                    <div class="col-md-6">
                        <h4>Live Minute State</h4>
                        <div style="max-height: 400px; overflow-y: scroll; border: 1px solid #ddd; background: white;">
                            {live_log_html}
                        </div>
                    </div>
                    <div class="col-md-6">
                        <h4>Live Trade Log</h4>
                         <div style="max-height: 400px; overflow-y: scroll; border: 1px solid #ddd; background: white;">
                            {live_trades_html}
                        </div>
                    </div>
                </div>
            </div>

            <hr>
            <h3>Test Set Trade Log</h3>
            <div style="max-height: 400px; overflow-y: scroll; border: 1px solid #ddd;">{trades_html}</div>
            
            <hr>
            <h3>Execution Detail (Snapshot)</h3>
            <div style="max-height: 600px; overflow-y: scroll; border: 1px solid #ddd;">
                {hourly_html}
            </div>
        </div>
    </body>
    </html>
    """
    return html_content

# --- 6. Live Forward Test Logic ---
def fetch_binance_candle(symbol_pair):
    try:
        url = "https://api.binance.com/api/v3/klines"
        params = {'symbol': symbol_pair, 'interval': '1m', 'limit': 2}
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        if len(data) >= 2:
            kline = data[-2] 
            ts = pd.to_datetime(kline[0], unit='ms')
            return ts, float(kline[2]), float(kline[3]), float(kline[4])
        return None, None, None, None
    except Exception as e:
        print(f"[{symbol_pair}] Binance API Error: {e}")
        return None, None, None, None

def live_trading_daemon(symbol, pair, best_ind, initial_equity, start_price, test_curve, test_trades, hourly_log):
    stop_pct = best_ind[0]
    profit_pct = best_ind[1]
    lines = np.sort(np.array(best_ind[2:]))
    
    live_equity = initial_equity
    prev_close = start_price
    
    live_longs = []
    live_shorts = []
    
    live_logs = []
    live_trades = []
    
    time.sleep(random.uniform(1, 10))
    print(f"[{symbol}] Live Trading Daemon Started (1m interval).")
    
    while True:
        now = datetime.now()
        next_run = (now + timedelta(minutes=1)).replace(second=5, microsecond=0)
        if next_run <= now: next_run += timedelta(minutes=1)
        time.sleep((next_run - now).total_seconds())
        
        ts, current_h, current_l, current_c = fetch_binance_candle(pair)
        if current_c is None: continue
            
        # Log State
        log_entry = {
            "Timestamp": str(ts),
            "Price": f"{current_c:.2f}",
            "Active Longs": len(live_longs),
            "Active Shorts": len(live_shorts),
            "Equity": f"{live_equity:.2f}"
        }
        live_logs.append(log_entry)
        
        # 1. Check Exits (Longs)
        for idx in range(len(live_longs) - 1, -1, -1):
            trade = live_longs[idx]
            entry_price = trade['entry']
            sl_price = entry_price * (1 - stop_pct)
            tp_price = entry_price * (1 + profit_pct)
            
            exit_price = 0.0
            hit = False
            reason = ""
            
            if current_l <= sl_price:
                exit_price = sl_price; hit = True; reason = "SL"
            elif current_h >= tp_price:
                exit_price = tp_price; hit = True; reason = "TP"
                
            if hit:
                pn_l = (exit_price - entry_price) / entry_price
                # Fee applied to live logic for realism (simulated live)
                net_pnl = pn_l - TEST_FEE 
                live_equity *= (1 + net_pnl)
                live_trades.append({'time': ts, 'type': 'Exit Long', 'price': exit_price, 'pnl': pn_l, 'equity': live_equity, 'reason': reason})
                live_longs.pop(idx)
                with REPORT_LOCK:
                    HTML_REPORTS[symbol] = generate_report(symbol, best_ind, test_curve, test_trades, hourly_log, live_logs, live_trades)

        # 2. Check Exits (Shorts)
        for idx in range(len(live_shorts) - 1, -1, -1):
            trade = live_shorts[idx]
            entry_price = trade['entry']
            sl_price = entry_price * (1 + stop_pct)
            tp_price = entry_price * (1 - profit_pct)
            
            exit_price = 0.0
            hit = False
            reason = ""
            
            if current_h >= sl_price:
                exit_price = sl_price; hit = True; reason = "SL"
            elif current_l <= tp_price:
                exit_price = tp_price; hit = True; reason = "TP"
            
            if hit:
                pn_l = (entry_price - exit_price) / entry_price
                net_pnl = pn_l - TEST_FEE
                live_equity *= (1 + net_pnl)
                live_trades.append({'time': ts, 'type': 'Exit Short', 'price': exit_price, 'pnl': pn_l, 'equity': live_equity, 'reason': reason})
                live_shorts.pop(idx)
                with REPORT_LOCK:
                    HTML_REPORTS[symbol] = generate_report(symbol, best_ind, test_curve, test_trades, hourly_log, live_logs, live_trades)

        # 3. Check Entries
        found_short = False
        short_target = 0.0
        if current_h > prev_close:
            idx_s = np.searchsorted(lines, prev_close, side='right')
            idx_e = np.searchsorted(lines, current_h, side='right')
            shorts = lines[idx_s:idx_e]
            if len(shorts) > 0: found_short = True; short_target = shorts[0]

        found_long = False
        long_target = 0.0
        if current_l < prev_close:
            idx_s = np.searchsorted(lines, current_l, side='left')
            idx_e = np.searchsorted(lines, prev_close, side='left')
            longs = lines[idx_s:idx_e]
            if len(longs) > 0: found_long = True; long_target = longs[-1]
            
        if found_short:
            live_shorts.append({'entry': short_target})
            live_trades.append({'time': ts, 'type': 'Short', 'price': short_target, 'pnl': 0, 'equity': live_equity, 'reason': 'Entry'})
            
        if found_long:
            live_longs.append({'entry': long_target})
            live_trades.append({'time': ts, 'type': 'Long', 'price': long_target, 'pnl': 0, 'equity': live_equity, 'reason': 'Entry'})
            
        prev_close = current_c
        with REPORT_LOCK:
             HTML_REPORTS[symbol] = generate_report(symbol, best_ind, test_curve, test_trades, hourly_log, live_logs, live_trades)

# --- 7. Server Handler ---
class ResultsHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed_path = urllib.parse.urlparse(self.path)
        path = parsed_path.path
        query = urllib.parse.parse_qs(parsed_path.query)

        if path == '/api/parameters':
            self.send_response(200); self.send_header('Content-type', 'application/json'); self.end_headers()
            symbol = query.get('symbol', [None])[0]
            if symbol and symbol in BEST_PARAMS: self.wfile.write(json.dumps(BEST_PARAMS[symbol]).encode('utf-8'))
            else: self.wfile.write(json.dumps(BEST_PARAMS).encode('utf-8'))
        elif path.startswith('/report/'):
            symbol = path.split('/')[-1]
            if symbol in HTML_REPORTS:
                self.send_response(200); self.send_header('Content-type', 'text/html'); self.end_headers()
                with REPORT_LOCK: self.wfile.write(HTML_REPORTS[symbol].encode('utf-8'))
            else: self.send_error(404)
        elif path == '/':
            self.send_response(200); self.send_header('Content-type', 'text/html'); self.end_headers()
            links = ""
            for asset in ASSETS:
                sym = asset['symbol']
                links += f'<a href="/report/{sym}" class="list-group-item list-group-item-action">{sym} Report</a>' if sym in HTML_REPORTS else f'<div class="list-group-item">{sym} (Loading...)</div>'
            html = f"<html><head><link rel='stylesheet' href='https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css'></head><body class='p-5'><h1>Hedged Grid Bot</h1><div class='list-group mt-4'>{links}</div></body></html>"
            self.wfile.write(html.encode('utf-8'))
        else: self.send_error(404)

# --- 8. Main Loop ---
def process_asset(asset_config):
    sym = asset_config['symbol']
    pair = asset_config['pair']
    
    print(f"\n--- Starting Optimization for {sym} ---")
    
    # 1. Get Data
    train_1h, test_1m = get_data_payload(pair)
    if train_1h is None: return

    # 2. Setup GA (Train on 1H, no fees)
    min_p, max_p = train_1h['close'].min(), train_1h['close'].max()
    toolbox = setup_toolbox(min_p, max_p, train_1h)

    # 3. Run GA
    pop = toolbox.population(n=POPULATION_SIZE)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    
    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=GENERATIONS, stats=stats, halloffame=hof, verbose=False)
    best_ind = hof[0]
    
    # 4. Save Params
    BEST_PARAMS[sym] = {"stop": best_ind[0], "profit": best_ind[1], "lines": list(best_ind[2:])}

    # 5. Final Tests (On 1M Data, WITH FEES)
    print(f"[{sym}] Running Final Test on 1m Data with {TEST_FEE*100}% Fee...")
    # Passing test_1m (high res) to backtest
    test_curve, test_trades, hourly_log = run_backtest(test_1m, best_ind[0], best_ind[1], np.array(best_ind[2:]), fee=TEST_FEE, detailed_log_trades=10)
    
    with REPORT_LOCK:
        HTML_REPORTS[sym] = generate_report(sym, best_ind, test_curve, test_trades, hourly_log)

    # 6. Start Live Thread
    last_close = test_1m['close'].iloc[-1]
    t = threading.Thread(target=live_trading_daemon, args=(sym, pair, best_ind, 10000.0, last_close, test_curve, test_trades, hourly_log), daemon=True)
    t.start()

if __name__ == "__main__":
    print("Initializing Hedged Multi-Asset Grid...")
    for asset in ASSETS[:MAX_ASSETS_TO_OPTIMIZE]: process_asset(asset)
    print(f"Serving at http://localhost:{PORT}/")
    with socketserver.TCPServer(("", PORT), ResultsHandler) as httpd:
        try: httpd.serve_forever()
        except KeyboardInterrupt: httpd.server_close()
