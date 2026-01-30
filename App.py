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
from datetime import datetime, timedelta
from deap import base, creator, tools, algorithms

# --- Configuration ---
PORT = 8080
N_LINES = 32
POPULATION_SIZE = 320
GENERATIONS = 10
RISK_FREE_RATE = 0.0
MAX_ASSETS_TO_OPTIMIZE = 1

# Costs
SLIPPAGE = 0.003  # 0.3%
FEE = 0.004       # 0.4%
# Total friction per side (Price impact + Fee)
FRICTION = SLIPPAGE + FEE

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

# --- 1. DEAP Initialization (Global Scope) ---
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# --- 2. Precise Data Ingestion (Binance 1 Month) ---
def fetch_binance_history(symbol_pair):
    base_url = "https://api.binance.com/api/v3/klines"
    
    # Calculate timestamps for the last 30 days
    end_time = datetime.now()
    start_time = end_time - timedelta(days=30)
    
    start_ts = int(start_time.timestamp() * 1000)
    end_ts = int(end_time.timestamp() * 1000)
    
    all_data = []
    current_start = start_ts
    
    print(f"[{symbol_pair}] Fetching last 30 days 1m data from Binance...")
    
    while current_start < end_ts:
        params = {
            'symbol': symbol_pair,
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
            
            # Update start time to the last timestamp + 1ms
            last_kline_ts = data[-1][0]
            current_start = last_kline_ts + 1
            
            # Safety break if we pass end time
            if last_kline_ts >= end_ts:
                break
                
            time.sleep(0.1) # Mild rate limit respect
            
        except Exception as e:
            print(f"CRITICAL API ERROR for {symbol_pair}: {e}")
            return None, None

    if not all_data:
        return None, None

    # Convert to DataFrame
    df = pd.DataFrame(all_data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 
        'close_time', 'quote_asset_volume', 'number_of_trades', 
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])
    
    df['dt'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['open'] = df['open'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['close'] = df['close'].astype(float)
    
    df.set_index('dt', inplace=True)
    df.sort_index(inplace=True)
    
    # Filter strictly last 30 days (in case API returned earlier context)
    df = df[df.index >= start_time]

    print(f"[{symbol_pair}] Raw 1m Data: {len(df)} rows")

    # Resample for GA speed
    df_1h = df.resample('1h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).dropna()

    print(f"[{symbol_pair}] Resampled 1H Data (For GA): {len(df_1h)} rows")
    
    if len(df_1h) < 50:
        print("Insufficient data after resampling.")
        return None, None

    split_idx = int(len(df_1h) * 0.85)
    train = df_1h.iloc[:split_idx]
    test = df_1h.iloc[split_idx:]
    
    return train, test

# --- 3. Strategy Logic (With Costs) ---
def run_backtest(df, stop_pct, profit_pct, lines, detailed_log_trades=0):
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    times = df.index
    
    equity = 10000.0
    equity_curve = [equity]
    position = 0          # 0: Flat, 1: Long, -1: Short
    entry_price = 0.0
    
    trades = []
    hourly_log = []
    
    lines = np.sort(lines)
    trades_completed = 0
    
    for i in range(1, len(df)):
        current_c = closes[i]
        current_h = highs[i]
        current_l = lows[i]
        prev_c = closes[i-1]
        ts = times[i]
        
        # --- Detailed Logging ---
        if detailed_log_trades > 0 and trades_completed < detailed_log_trades:
            idx = np.searchsorted(lines, current_c)
            val_below = lines[idx-1] if idx > 0 else -999.0
            val_above = lines[idx] if idx < len(lines) else 999999.0
            
            act_sl = np.nan
            act_tp = np.nan
            pos_str = "FLAT"
            
            if position == 1:
                pos_str = "LONG"
                act_sl = entry_price * (1 - stop_pct)
                act_tp = entry_price * (1 + profit_pct)
            elif position == -1:
                pos_str = "SHORT"
                act_sl = entry_price * (1 + stop_pct)
                act_tp = entry_price * (1 - profit_pct)
            
            log_entry = {
                "Timestamp": str(ts),
                "Price": f"{current_c:.2f}",
                "Nearest Below": f"{val_below:.2f}" if val_below != -999 else "None",
                "Nearest Above": f"{val_above:.2f}" if val_above != 999999 else "None",
                "Position": pos_str,
                "Active SL": f"{act_sl:.2f}" if not np.isnan(act_sl) else "-",
                "Active TP": f"{act_tp:.2f}" if not np.isnan(act_tp) else "-",
                "Equity": f"{equity:.2f}"
            }
            hourly_log.append(log_entry)

        # --- Strategy Execution ---
        if position != 0:
            sl_hit = False
            tp_hit = False
            exit_price = 0.0
            reason = ""

            if position == 1: # Long Logic
                sl_price = entry_price * (1 - stop_pct)
                tp_price = entry_price * (1 + profit_pct)
                
                if current_l <= sl_price:
                    sl_hit = True; exit_price = sl_price 
                elif current_h >= tp_price:
                    tp_hit = True; exit_price = tp_price

            elif position == -1: # Short Logic
                sl_price = entry_price * (1 + stop_pct)
                tp_price = entry_price * (1 - profit_pct)
                
                if current_h >= sl_price:
                    sl_hit = True; exit_price = sl_price
                elif current_l <= tp_price:
                    tp_hit = True; exit_price = tp_price
            
            if sl_hit or tp_hit:
                # APPLY EXIT COSTS
                # Slippage worsens exit price, Fee reduces PnL
                if position == 1: 
                    # Sell: Price decreases by slippage
                    effective_exit = exit_price * (1 - SLIPPAGE)
                    # Buy was: entry_price * (1 + SLIPPAGE) -- Applied at entry
                    gross_pnl = (effective_exit - (entry_price * (1 + SLIPPAGE))) / (entry_price * (1 + SLIPPAGE))
                    # Deduct Fee (approximate on notional)
                    net_pnl = gross_pnl - (FEE * 2) # Entry fee + Exit fee
                    
                else: 
                    # Buy to cover: Price increases by slippage
                    effective_exit = exit_price * (1 + SLIPPAGE)
                    # Sell was: entry_price * (1 - SLIPPAGE)
                    gross_pnl = ((entry_price * (1 - SLIPPAGE)) - effective_exit) / (entry_price * (1 - SLIPPAGE))
                    net_pnl = gross_pnl - (FEE * 2)

                equity *= (1 + net_pnl)
                reason = "SL" if sl_hit else "TP"
                trades.append({'time': ts, 'type': 'Exit', 'price': exit_price, 'pnl': net_pnl, 'equity': equity, 'reason': reason})
                position = 0
                trades_completed += 1
                equity_curve.append(equity)
                continue 

        if position == 0:
            found_short = False
            short_price = 0.0
            
            # Check for lines between prev_c and current_h (Short Candidates)
            if current_h > prev_c:
                idx_s = np.searchsorted(lines, prev_c, side='right')   
                idx_e = np.searchsorted(lines, current_h, side='right') 
                potential_shorts = lines[idx_s:idx_e]
                
                if len(potential_shorts) > 0:
                    found_short = True
                    short_price = potential_shorts[0] 

            found_long = False
            long_price = 0.0
            
            # Check for lines between current_l and prev_c (Long Candidates)
            if current_l < prev_c:
                idx_s = np.searchsorted(lines, current_l, side='left') 
                idx_e = np.searchsorted(lines, prev_c, side='left')    
                potential_longs = lines[idx_s:idx_e]
                
                if len(potential_longs) > 0:
                    found_long = True
                    long_price = potential_longs[-1] 

            # Execution Decision
            target_line = 0.0
            new_pos = 0
            
            if found_short and found_long:
                if current_c > prev_c:
                    new_pos = -1; target_line = short_price
                else:
                    new_pos = 1; target_line = long_price
            elif found_short:
                new_pos = -1; target_line = short_price
            elif found_long:
                new_pos = 1; target_line = long_price
            
            if new_pos != 0:
                position = new_pos
                entry_price = target_line
                # Entry costs are realized at Exit calculation or effectively embedded in PnL logic above
                trades.append({'time': ts, 'type': 'Short' if position == -1 else 'Long', 'price': entry_price, 'pnl': 0, 'equity': equity, 'reason': 'Entry'})

        equity_curve.append(equity)

    return equity_curve, trades, hourly_log

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
    eq_curve, _, _ = run_backtest(df_train, stop_pct, profit_pct, lines, detailed_log_trades=0)
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
def generate_report(symbol, best_ind, train_data, test_data, train_curve, test_curve, test_trades, hourly_log, live_logs=[], live_trades=[]):
    plt.figure(figsize=(14, 12))
    
    # 1. Equity Curve
    plt.subplot(2, 1, 1)
    plt.title(f"{symbol} Equity Curve: Training (Blue) vs Test (Orange) - 30 Days")
    plt.plot(train_curve, label='Training Equity')
    plt.plot(range(len(train_curve), len(train_curve)+len(test_curve)), test_curve, label='Test Equity')
    plt.legend()
    plt.grid(True)
    
    # 2. Full Price Action
    plt.subplot(2, 1, 2)
    plt.title(f"{symbol} Test Set Price Action & Grid Lines")
    plt.plot(test_data.index, test_data['close'], color='black', alpha=1, label='Price', linewidth=0.8)
    
    lines = best_ind[2:]
    min_test = test_data['low'].min()
    max_test = test_data['high'].max()
    margin = (max_test - min_test) * 0.1
    visible_lines = [l for l in lines if (min_test - margin) < l < (max_test + margin)]
    
    for l in visible_lines:
        plt.axhline(y=l, color='blue', alpha=0.1, linewidth=0.5)
    
    plt.tight_layout()
    
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
        <li class="list-group-item"><strong>Active Grid Lines:</strong> {N_LINES}</li>
        <li class="list-group-item"><strong>Friction (Fee+Slip):</strong> {(FRICTION*2)*100:.2f}% / Trade</li>
        <li class="list-group-item"><a href="/api/parameters?symbol={symbol}" target="_blank">View JSON Parameters</a></li>
    </ul>
    """

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{symbol} Strategy Results</title>
        <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
        <meta http-equiv="refresh" content="3000"> 
        <style>body {{ padding: 20px; }} h3 {{ margin-top: 30px; }} th {{ position: sticky; top: 0; background: white; }}</style>
    </head>
    <body>
        <div class="container-fluid">
            <a href="/" class="btn btn-secondary mb-3">&larr; Back to Dashboard</a>
            <h1 class="mb-4">{symbol} Grid Strategy GA Results (Binance 30d)</h1>
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
                <p><strong>Status:</strong> Running. Fetches candle at XX:XX:05 (Every Minute).</p>
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
            <h3>Trade Log (Test Set)</h3>
            <div style="max-height: 400px; overflow-y: scroll; border: 1px solid #ddd;">{trades_html}</div>
            
            <hr>
            <h3>Hourly Details (First 5 Trades Timeline)</h3>
            <div style="max-height: 600px; overflow-y: scroll; border: 1px solid #ddd;">
                {hourly_html}
            </div>
        </div>
    </body>
    </html>
    """
    return html_content

# --- 6. Live Forward Test Logic (1 Minute Update) ---
def fetch_binance_candle(symbol_pair):
    try:
        url = "https://api.binance.com/api/v3/klines"
        params = {
            'symbol': symbol_pair,
            'interval': '1m', 
            'limit': 2 
        }
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        if len(data) >= 2:
            kline = data[-2] 
            ts = pd.to_datetime(kline[0], unit='ms')
            high_price = float(kline[2])
            low_price = float(kline[3])
            close_price = float(kline[4])
            return ts, high_price, low_price, close_price
        return None, None, None, None
    except Exception as e:
        print(f"[{symbol_pair}] Binance API Error: {e}")
        return None, None, None, None

def live_trading_daemon(symbol, pair, best_ind, initial_equity, start_price, train_df, test_df, train_curve, test_curve, test_trades, hourly_log):
    
    stop_pct = best_ind[0]
    profit_pct = best_ind[1]
    lines = np.sort(np.array(best_ind[2:]))
    
    live_equity = initial_equity
    live_position = 0 
    live_entry_price = 0.0
    prev_close = start_price
    
    live_logs = []
    live_trades = []
    
    time.sleep(random.uniform(1, 10))
    print(f"[{symbol}] Live Trading Daemon Started (1m interval).")
    
    while True:
        now = datetime.now()
        next_run = (now + timedelta(minutes=1)).replace(second=5, microsecond=0)
        
        if next_run <= now:
            next_run += timedelta(minutes=1)
            
        sleep_sec = (next_run - now).total_seconds()
        sleep_sec += random.uniform(0.1, 1.0)
        
        time.sleep(sleep_sec)
        
        ts, current_h, current_l, current_c = fetch_binance_candle(pair)
        
        if current_c is None:
           
