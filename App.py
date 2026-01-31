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
TRAIN_SPLIT = 0.90
PORT = 8080

# Trading Costs
FEE = 0.002
SLIPPAGE = 0.001
STOP_LOSS = 0.005
TAKE_PROFIT = 0.03

# GA Parameters
POP_SIZE = 50
GENERATIONS = 20
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.7
MAX_LEVELS = 100
MIN_LEVELS = 5

def fetch_binance_data(symbol="BTCUSDT", interval="1h", start_str="2020-01-01"):
    """
    Fetches historical OHLC data from Binance API.
    """
    base_url = "https://api.binance.com/api/v3/klines"
    
    # Convert start string to ms timestamp
    start_dt = datetime.strptime(start_str, "%Y-%m-%d")
    start_ts = int(start_dt.timestamp() * 1000)
    
    # End time is now
    end_ts = int(time.time() * 1000)
    
    all_data = []
    current_start = start_ts
    
    print(f"Fetching {symbol} {interval} data from {start_str}...")
    
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
            
            # Update start time: Last candle open time + 1 interval (1h = 3600000ms)
            last_open_time = data[-1][0]
            current_start = last_open_time + 3600000
            
            # Progress indicator
            last_date = datetime.fromtimestamp(last_open_time / 1000)
            print(f"Fetched up to {last_date}...", end='\r')
            
            if current_start > end_ts:
                break
                
            # Rate limit protection (Binance is generous, but safe side)
            time.sleep(0.1)
            
        except Exception as e:
            print(f"\nError fetching data: {e}")
            break
            
    print(f"\nDownload complete. Total candles: {len(all_data)}")
    
    # Binance Columns: 
    # 0: Open time, 1: Open, 2: High, 3: Low, 4: Close, 5: Volume, ...
    df = pd.DataFrame(all_data, columns=[
        "open_time", "open", "high", "low", "close", "volume", 
        "close_time", "q_vol", "trades", "tb_base", "tb_quote", "ignore"
    ])
    
    # Type conversion
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
        self.n = len(df)

    def run(self, price_levels):
        position = 0
        entry_price = 0.0
        equity = [10000.0]
        trades = []
        
        if len(price_levels) == 0:
            return equity, trades

        levels = np.array(price_levels)
        
        for t in range(1, self.n):
            current_equity = equity[-1]
            h = self.high[t]
            l = self.low[t]
            o = self.open[t]
            prev_c = self.close[t-1]
            
            if position != 0:
                pnl = 0.0
                executed = False
                
                if position == 1:
                    sl_price = entry_price * (1 - STOP_LOSS)
                    tp_price = entry_price * (1 + TAKE_PROFIT)
                    
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
                    sl_price = entry_price * (1 + STOP_LOSS)
                    tp_price = entry_price * (1 - TAKE_PROFIT)
                    
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
            
            if position == 0:
                relevant_levels = levels[(levels >= l) & (levels <= h)]
                
                if len(relevant_levels) > 0:
                    for lvl in relevant_levels:
                        if prev_c < lvl <= h:
                            fill_price = lvl
                            position = -1
                            entry_price = fill_price * (1 - SLIPPAGE) 
                            current_equity = current_equity * (1 - FEE) 
                            break 
                        
                        elif prev_c > lvl >= l:
                            fill_price = lvl
                            position = 1
                            entry_price = fill_price * (1 + SLIPPAGE)
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
        
        self.min_price = data_train['low'].min()
        self.max_price = data_train['high'].max()
        
        self.population = np.random.rand(pop_size, MAX_LEVELS + 1)
        
    def decode_chromosome(self, chrom):
        count_norm = chrom[0]
        count = int(MIN_LEVELS + count_norm * (MAX_LEVELS - MIN_LEVELS))
        levels_norm = chrom[1:]
        
        raw_levels = levels_norm[:count]
        prices = self.min_price + raw_levels * (self.max_price - self.min_price)
        prices.sort()
        return prices

    def fitness(self, chrom):
        levels = self.decode_chromosome(chrom)
        equity, _ = self.backtester.run(levels)
        
        if len(equity) == 0 or equity[-1] == 10000.0:
            return -1.0
            
        ret = (equity[-1] - 10000.0) / 10000.0
        return ret

    def evolve(self):
        print(f"Starting Optimization: {self.generations} Generations, Pop {self.pop_size}")
        
        for gen in range(self.generations):
            scores = []
            for i in range(self.pop_size):
                scores.append(self.fitness(self.population[i]))
            
            scores = np.array(scores)
            best_idx = np.argmax(scores)
            print(f"Gen {gen+1}/{self.generations} | Best Return: {scores[best_idx]*100:.2f}%")
            
            new_pop = np.zeros_like(self.population)
            new_pop[0] = self.population[best_idx]
            
            for i in range(1, self.pop_size):
                candidates = np.random.choice(self.pop_size, 3)
                parent1_idx = candidates[np.argmax(scores[candidates])]
                candidates = np.random.choice(self.pop_size, 3)
                parent2_idx = candidates[np.argmax(scores[candidates])]
                
                parent1 = self.population[parent1_idx]
                parent2 = self.population[parent2_idx]
                
                if np.random.rand() < CROSSOVER_RATE:
                    cross_point = np.random.randint(1, len(parent1))
                    child = np.concatenate((parent1[:cross_point], parent2[cross_point:]))
                else:
                    child = parent1.copy()
                
                mutation_mask = np.random.rand(len(child)) < MUTATION_RATE
                random_genes = np.random.rand(len(child))
                child[mutation_mask] = random_genes[mutation_mask]
                
                new_pop[i] = child
            
            self.population = new_pop
            
        final_scores = [self.fitness(p) for p in self.population]
        best_chrom = self.population[np.argmax(final_scores)]
        return best_chrom

def plot_candlesticks(df, levels, filename="ohlc.png"):
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

    xmin, xmax = 0, len(df_plot)
    for level in levels:
        plt.hlines(level, xmin, xmax, colors='blue', linestyles='dashed', alpha=0.6, linewidth=0.8)

    plt.title('Test Data: OHLC & Optimized Price Levels')
    plt.xlabel('Time (Candles)')
    plt.ylabel('Price')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def main():
    # 1. Fetch Data from Binance
    df = fetch_binance_data(symbol="BTCUSDT", interval="1h", start_str="2020-01-01")
    
    # 2. Split Data
    split_idx = int(len(df) * TRAIN_SPLIT)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    print(f"Data loaded. Train size: {len(train_df)}, Test size: {len(test_df)}")
    
    # 3. Optimize (GA)
    optimizer = GeneticOptimizer(train_df)
    best_chrom = optimizer.evolve()
    best_levels = optimizer.decode_chromosome(best_chrom)
    
    print(f"Optimization complete. Best configuration has {len(best_levels)} levels.")
    
    # 4. Test on Out-of-Sample Data
    bt = Backtester(test_df)
    equity, trades = bt.run(best_levels)
    
    # 5. Generate Reports
    os.makedirs("output", exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    plt.plot(equity, label='Equity Curve (Test Data)')
    plt.title(f'Strategy Performance (Test Data)\nLevels: {len(best_levels)}')
    plt.xlabel('Candles')
    plt.ylabel('Capital ($)')
    plt.legend()
    plt.grid(True)
    plt.savefig('output/equity.png')
    plt.close()

    print("Generating OHLC plot...")
    plot_candlesticks(test_df, best_levels, "output/ohlc.png")
    
    total_return = (equity[-1] - 10000) / 10000 * 100
    win_rate = len([t for t in trades if t > 0]) / len(trades) * 100 if trades else 0
    
    html_content = f"""
    <html>
    <head><title>Trading Strategy Report</title></head>
    <body style="font-family: monospace; padding: 20px;">
        <h1>Optimization Results</h1>
        <hr>
        <h2>Metrics (Test Set 10%)</h2>
        <ul>
            <li>Initial Capital: $10,000</li>
            <li>Final Capital: ${equity[-1]:.2f}</li>
            <li>Total Return: {total_return:.2f}%</li>
            <li>Total Trades: {len(trades)}</li>
            <li>Win Rate: {win_rate:.2f}%</li>
            <li>Optimized Level Count: {len(best_levels)}</li>
        </ul>
        <hr>
        <h2>Equity Curve</h2>
        <img src="equity.png" style="max-width: 100%; border: 1px solid #ddd;">
        <hr>
        <h2>Price Action & Levels</h2>
        <img src="ohlc.png" style="max-width: 100%; border: 1px solid #ddd;">
        <hr>
        <h2>Price Levels</h2>
        <p>{list(np.round(best_levels, 2))}</p>
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
