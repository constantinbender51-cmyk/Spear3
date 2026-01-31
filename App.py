import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import io
import os
import http.server
import socketserver
import time
from datetime import datetime
import random

# --- Configuration ---
TRAIN_SPLIT = 0.70  # Modified to 70/30
PORT = 8080

# Trading Costs (Fixed)
FEE = 0.002
SLIPPAGE = 0.001

# GA Parameters
POP_SIZE = 100
GENERATIONS = 40
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.7

# Strategy Constraints
NUM_LEVELS = 50 

# Optimization Bounds
MIN_SL = 0.001   # 0.1%
MAX_SL = 0.05    # 5.0%
MIN_TP = 0.005   # 0.5%
MAX_TP = 0.10    # 10.0%

# SMA Offset Bounds (Percentage from SMA)
MIN_OFFSET = -0.20  # -20% from SMA
MAX_OFFSET = 0.20   # +20% from SMA

def fetch_binance_data(symbol="BTCUSDT", interval="1h", start_str="2020-01-01"):
    base_url = "https://api.binance.com/api/v3/klines"
    start_dt = datetime.strptime(start_str, "%Y-%m-%d")
    start_ts = int(start_dt.timestamp() * 1000)
    end_ts = int(time.time() * 1000)
    
    all_data = []
    current_start = start_ts
    
    print(f"Fetching {symbol} {interval} data from {start_str}...")
    
    # Calculate ms per interval for pagination
    interval_map = {"1m": 60000, "30m": 1800000, "1h": 3600000}
    step_ms = interval_map.get(interval, 3600000)

    while True:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": current_start,
            "limit": 1000 
        }
        
        try:
            r = requests.get(base_url, params=params)
            r.raise_for_status()
            data = r.json()
            
            if not data:
                break
                
            all_data.extend(data)
            last_open_time = data[-1][0]
            current_start = last_open_time + step_ms
            
            last_date = datetime.fromtimestamp(last_open_time / 1000)
            print(f"Fetched up to {last_date}...", end='\r')
            
            if current_start > end_ts:
                break
            time.sleep(0.1)
            
        except Exception as e:
            print(f"\nError fetching data: {e}")
            break
            
    print(f"\nDownload complete. Total candles: {len(all_data)}")
    
    df = pd.DataFrame(all_data, columns=[
        "open_time", "open", "high", "low", "close", "volume", 
        "close_time", "q_vol", "trades", "tb_base", "tb_quote", "ignore"
    ])
    
    cols = ["open", "high", "low", "close", "volume"]
    for c in cols:
        df[c] = df[c].astype(float)
        
    return df

class Backtester:
    def __init__(self, df):
        self.df = df.copy().reset_index(drop=True)
        self.high = self.df['high'].values
        self.low = self.df['low'].values
        self.open = self.df['open'].values
        self.close = self.df['close'].values
        self.sma = self.df['sma'].values  # Load SMA
        self.n = len(df)

    def run(self, level_offsets, stop_loss, take_profit):
        """
        Executes the strategy with dynamic levels based on SMA offsets.
        Direction: Breakout (Same direction as touch).
        """
        position = 0
        entry_price = 0.0
        equity = [10000.0]
        trades = []
        
        if len(level_offsets) == 0:
            return equity, trades

        offsets = np.array(level_offsets)
        
        for t in range(1, self.n):
            current_equity = equity[-1]
            h = self.high[t]
            l = self.low[t]
            prev_c = self.close[t-1]
            current_sma = self.sma[t]
            
            # --- Check Exit ---
            if position != 0:
                pnl = 0.0
                executed = False
                
                if position == 1:
                    sl_price = entry_price * (1 - stop_loss)
                    tp_price = entry_price * (1 + take_profit)
                    
                    if l <= sl_price:
                        exit_price = sl_price * (1 - SLIPPAGE)
                        pnl = (exit_price - entry_price) / entry_price
                        pnl -= FEE 
                        executed = True
                    elif h >= tp_price:
                        exit_price = tp_price * (1 - SLIPPAGE)
                        pnl = (exit_price - entry_price) / entry_price
                        pnl -= FEE
                        executed = True
                
                elif position == -1:
                    sl_price = entry_price * (1 + stop_loss)
                    tp_price = entry_price * (1 - take_profit)
                    
                    if h >= sl_price:
                        exit_price = sl_price * (1 + SLIPPAGE)
                        pnl = (entry_price - exit_price) / entry_price
                        pnl -= FEE
                        executed = True
                    elif l <= tp_price:
                        exit_price = tp_price * (1 + SLIPPAGE)
                        pnl = (entry_price - exit_price) / entry_price
                        pnl -= FEE
                        executed = True

                if executed:
                    new_equity = current_equity * (1 + pnl)
                    equity.append(new_equity)
                    trades.append(pnl)
                    position = 0
                    entry_price = 0.0
                    continue 
            
            # --- Check Entry ---
            if position == 0:
                # Calculate dynamic levels for current timestamp
                # level = SMA * (1 + offset)
                current_levels = current_sma * (1 + offsets)
                
                # Filter relevant levels strictly within high/low
                # Optimization: Vector filtering
                mask = (current_levels >= l) & (current_levels <= h)
                
                if np.any(mask):
                    relevant_levels = current_levels[mask]
                    for lvl in relevant_levels:
                        if prev_c < lvl <= h:
                            # Cross UP -> LONG (Breakout)
                            fill_price = lvl
                            position = 1
                            entry_price = fill_price * (1 + SLIPPAGE)
                            current_equity = current_equity * (1 - FEE)
                            break 
                        
                        elif prev_c > lvl >= l:
                            # Cross DOWN -> SHORT (Breakdown)
                            fill_price = lvl
                            position = -1
                            entry_price = fill_price * (1 - SLIPPAGE) 
                            current_equity = current_equity * (1 - FEE) 
                            break 
            
            equity.append(current_equity)
            
        return equity, trades

class GeneticOptimizer:
    def __init__(self, data_train, pop_size=POP_SIZE, generations=GENERATIONS):
        self.data = data_train
        self.pop_size = pop_size
        self.generations = generations
        self.backtester = Backtester(data_train)
        
        # Chromosome Structure:
        # 0: Stop Loss (Normalized)
        # 1: Take Profit (Normalized)
        # 2 to 51: Price Level Offsets (Normalized, Fixed Count 50)
        self.gene_length = 2 + NUM_LEVELS
        self.population = np.random.rand(pop_size, self.gene_length)
        
    def decode_chromosome(self, chrom):
        # 1. Decode SL
        sl_norm = chrom[0]
        stop_loss = MIN_SL + sl_norm * (MAX_SL - MIN_SL)
        
        # 2. Decode TP
        tp_norm = chrom[1]
        take_profit = MIN_TP + tp_norm * (MAX_TP - MIN_TP)
        
        # 3. Decode Level Offsets (Percentage from SMA)
        # Maps 0..1 to MIN_OFFSET..MAX_OFFSET
        offsets_norm = chrom[2:]
        offsets = MIN_OFFSET + offsets_norm * (MAX_OFFSET - MIN_OFFSET)
        offsets.sort()
        
        return offsets, stop_loss, take_profit

    def fitness(self, chrom):
        offsets, sl, tp = self.decode_chromosome(chrom)
        equity, trades = self.backtester.run(offsets, sl, tp)
        
        # Penalize insufficient data points or bankruptcy
        if len(trades) < 2 or equity[-1] <= 100.0:
            return -10.0
        
        # Calculate Returns
        equity_curve = np.array(equity)
        # Avoid division by zero in returns calculation if equity is zero (already caught above, but safety first)
        returns = np.diff(equity_curve) / equity_curve[:-1]
        
        std_dev = np.std(returns)
        if std_dev < 1e-9:
            return -10.0
        
        # Sharpe Ratio Calculation
        # Annualization factor for 1h candles: sqrt(24 * 365) = 93.59
        annualization_factor = np.sqrt(24 * 365)
        sharpe = (np.mean(returns) / std_dev) * annualization_factor
        
        return sharpe

    def evolve(self):
        print(f"Starting Optimization: {self.generations} Gens, Pop {self.pop_size}, Levels {NUM_LEVELS}")
        print(f"Optimizing for Sharpe Ratio (1h timeframe)")
        print(f"Ranges: SL ({MIN_SL*100}%-{MAX_SL*100}%) | TP ({MIN_TP*100}%-{MAX_TP*100}%)")
        print(f"SMA Offsets: {MIN_OFFSET*100}% to {MAX_OFFSET*100}%")
        
        for gen in range(self.generations):
            scores = []
            for i in range(self.pop_size):
                scores.append(self.fitness(self.population[i]))
            
            scores = np.array(scores)
            best_idx = np.argmax(scores)
            best_sharpe = scores[best_idx]
            
            # Extract current best params for logging
            _, b_sl, b_tp = self.decode_chromosome(self.population[best_idx])
            print(f"Gen {gen+1}/{self.generations} | Sharpe: {best_sharpe:.4f} | SL: {b_sl*100:.2f}% | TP: {b_tp*100:.2f}%")
            
            new_pop = np.zeros_like(self.population)
            # Elitism
            new_pop[0] = self.population[best_idx]
            
            for i in range(1, self.pop_size):
                # Tournament Selection
                cands = np.random.choice(self.pop_size, 3)
                p1_idx = cands[np.argmax(scores[cands])]
                cands = np.random.choice(self.pop_size, 3)
                p2_idx = cands[np.argmax(scores[cands])]
                
                parent1 = self.population[p1_idx]
                parent2 = self.population[p2_idx]
                
                # Crossover
                if np.random.rand() < CROSSOVER_RATE:
                    cross_point = np.random.randint(1, self.gene_length)
                    child = np.concatenate((parent1[:cross_point], parent2[cross_point:]))
                else:
                    child = parent1.copy()
                
                # Mutation
                mutation_mask = np.random.rand(self.gene_length) < MUTATION_RATE
                random_genes = np.random.rand(self.gene_length)
                child[mutation_mask] = random_genes[mutation_mask]
                
                new_pop[i] = child
            
            self.population = new_pop
            
        final_scores = [self.fitness(p) for p in self.population]
        best_chrom = self.population[np.argmax(final_scores)]
        return best_chrom

def plot_candlesticks(df, filename="ohlc.png"):
    plt.figure(figsize=(14, 8))
    df_plot = df.reset_index(drop=True)
    
    up_color = '#2ca02c'
    down_color = '#d62728'
    width = 0.6
    width2 = 0.05
    
    up = df_plot[df_plot.close >= df_plot.open]
    down = df_plot[df_plot.close < df_plot.open]
    
    plt.bar(up.index, up.close - up.open, width, bottom=up.open, color=up_color)
    plt.bar(up.index, up.high - up.close, width2, bottom=up.close, color=up_color)
    plt.bar(up.index, up.low - up.open, width2, bottom=up.open, color=up_color)
    
    plt.bar(down.index, down.open - down.close, width, bottom=down.close, color=down_color)
    plt.bar(down.index, down.high - down.open, width2, bottom=down.open, color=down_color)
    plt.bar(down.index, down.low - down.close, width2, bottom=down.close, color=down_color)

    # Plot SMA
    if 'sma' in df_plot.columns:
        plt.plot(df_plot.index, df_plot['sma'], color='blue', label='SMA 365', linewidth=1.5)

    plt.title('Test Data: OHLC & SMA 365')
    plt.xlabel('Time (Candles)')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def main():
    # 1. Fetch Data
    df = fetch_binance_data(symbol="BTCUSDT", interval="1h", start_str="2020-01-01")
    
    # 2. Preprocess SMA
    print("Calculating SMA 365...")
    df['sma'] = df['close'].rolling(window=365).mean()
    # Drop initial NaNs
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # 3. Split Data
    split_idx = int(len(df) * TRAIN_SPLIT)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    print(f"Data prepared. Train size: {len(train_df)}, Test size: {len(test_df)}")
    
    # 4. Optimize (GA)
    optimizer = GeneticOptimizer(train_df)
    best_chrom = optimizer.evolve()
    best_offsets, best_sl, best_tp = optimizer.decode_chromosome(best_chrom)
    
    print(f"Optimization complete.")
    print(f"Best SL: {best_sl*100:.2f}%")
    print(f"Best TP: {best_tp*100:.2f}%")
    
    # 5. Test on Out-of-Sample Data
    bt = Backtester(test_df)
    equity, trades = bt.run(best_offsets, best_sl, best_tp)
    
    # 6. Generate Reports
    os.makedirs("output", exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    plt.plot(equity, label='Equity Curve (Test Data)')
    plt.title(f'Strategy Performance (Test Data)\nSMA Offsets: {len(best_offsets)} | SL: {best_sl*100:.2f}% | TP: {best_tp*100:.2f}%')
    plt.xlabel('Candles')
    plt.ylabel('Capital ($)')
    plt.legend()
    plt.grid(True)
    plt.savefig('output/equity.png')
    plt.close()

    print("Generating OHLC plot...")
    plot_candlesticks(test_df, "output/ohlc.png")
    
    total_return = (equity[-1] - 10000) / 10000 * 100
    win_rate = len([t for t in trades if t > 0]) / len(trades) * 100 if trades else 0
    
    html_content = f"""
    <html>
    <head><title>Trading Strategy Report</title></head>
    <body style="font-family: monospace; padding: 20px;">
        <h1>Optimization Results</h1>
        <hr>
        <h2>Metrics (Test Set {100-TRAIN_SPLIT*100:.0f}%)</h2>
        <ul>
            <li>Initial Capital: $10,000</li>
            <li>Final Capital: ${equity[-1]:.2f}</li>
            <li>Total Return: {total_return:.2f}%</li>
            <li>Total Trades: {len(trades)}</li>
            <li>Win Rate: {win_rate:.2f}%</li>
            <li>Optimized SL: {best_sl*100:.2f}%</li>
            <li>Optimized TP: {best_tp*100:.2f}%</li>
        </ul>
        <hr>
        <h2>Equity Curve</h2>
        <img src="equity.png" style="max-width: 100%; border: 1px solid #ddd;">
        <hr>
        <h2>Price Action & SMA</h2>
        <img src="ohlc.png" style="max-width: 100%; border: 1px solid #ddd;">
        <hr>
        <h2>SMA Offset Levels (%)</h2>
        <p>{list(np.round(best_offsets * 100, 2))}</p>
    </body>
    </html>
    """
    
    with open("output/index.html", "w") as f:
        f.write(html_content)
        
    print(f"Results saved to output/. Serving on port {PORT}...")
    
    os.chdir("output")
    with socketserver.TCPServer(("", PORT), http.server.SimpleHTTPRequestHandler) as httpd:
        print(f"Serving at http://localhost:{PORT}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            httpd.server_close()
            print("Server stopped.")

if __name__ == "__main__":
    main()
