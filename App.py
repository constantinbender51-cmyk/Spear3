import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import io
import os
import http.server
import socketserver
from datetime import datetime
import random

# --- Configuration ---
DATA_URL = "https://ohlcendpoint.up.railway.app/data/btc1h.csv"
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

class Backtester:
    def __init__(self, df):
        self.df = df.copy().reset_index(drop=True)
        self.high = self.df['high'].values
        self.low = self.df['low'].values
        self.open = self.df['open'].values
        self.close = self.df['close'].values
        self.n = len(df)

    def run(self, price_levels):
        """
        Executes the strategy.
        price_levels: sorted list of active price levels.
        """
        # State
        position = 0  # 0: Flat, 1: Long, -1: Short
        entry_price = 0.0
        equity = [10000.0]  # Start with 10k
        trades = []
        
        # Pre-calculation for speed
        if len(price_levels) == 0:
            return equity, trades

        levels = np.array(price_levels)
        
        for t in range(1, self.n):
            current_equity = equity[-1]
            h = self.high[t]
            l = self.low[t]
            o = self.open[t]
            prev_c = self.close[t-1]
            
            # --- Check Exit (if position exists) ---
            if position != 0:
                pnl = 0.0
                executed = False
                
                # Long Exit Logic
                if position == 1:
                    sl_price = entry_price * (1 - STOP_LOSS)
                    tp_price = entry_price * (1 + TAKE_PROFIT)
                    
                    # Check worst case: SL hit first if both triggered (Conservative)
                    if l <= sl_price:
                        exit_price = sl_price * (1 - SLIPPAGE) # Sell order slippage
                        pnl = (exit_price - entry_price) / entry_price
                        # Fee on exit
                        pnl -= FEE 
                        executed = True
                    elif h >= tp_price:
                        exit_price = tp_price * (1 - SLIPPAGE)
                        pnl = (exit_price - entry_price) / entry_price
                        pnl -= FEE
                        executed = True
                
                # Short Exit Logic
                elif position == -1:
                    sl_price = entry_price * (1 + STOP_LOSS)
                    tp_price = entry_price * (1 - TAKE_PROFIT)
                    
                    if h >= sl_price:
                        exit_price = sl_price * (1 + SLIPPAGE) # Buy order slippage
                        pnl = (entry_price - exit_price) / entry_price
                        pnl -= FEE
                        executed = True
                    elif l <= tp_price:
                        exit_price = tp_price * (1 + SLIPPAGE)
                        pnl = (entry_price - exit_price) / entry_price
                        pnl -= FEE
                        executed = True

                if executed:
                    # Apply PnL
                    new_equity = current_equity * (1 + pnl)
                    equity.append(new_equity)
                    trades.append(pnl)
                    position = 0
                    entry_price = 0.0
                    continue # Wait for next candle for new entry (simplify logic)
            
            # --- Check Entry (if flat) ---
            if position == 0:
                # Find relevant levels
                # Short: Touch from below (Prev Close < Level <= High)
                # Long: Touch from above (Prev Close > Level >= Low)
                
                # Optimization: Only check levels within candle range
                relevant_levels = levels[(levels >= l) & (levels <= h)]
                
                if len(relevant_levels) > 0:
                    # Prioritize the one closest to Open or first touched? 
                    # Assuming limit orders exist, the first one hit is the one closest to Open.
                    
                    # Logic: 
                    # If Short: Price goes UP to level. Level > Prev_Close.
                    # If Long: Price goes DOWN to level. Level < Prev_Close.
                    
                    for lvl in relevant_levels:
                        if prev_c < lvl <= h:
                            # Short Trigger
                            # Entry Price calculation
                            fill_price = lvl
                            
                            # Execute Short
                            # Sell at fill_price * (1 - slippage)
                            # Fee deducted immediately from potential
                            position = -1
                            # Effective cost basis accounting for slippage/fees in the PnL calculation later
                            # For tracking "entry_price" we use the raw level, 
                            # but we account for slippage/fee immediately on equity or at exit?
                            # Standard: Record executed price.
                            entry_price = fill_price * (1 - SLIPPAGE) 
                            # Pay entry fee
                            current_equity = current_equity * (1 - FEE) 
                            break 
                        
                        elif prev_c > lvl >= l:
                            # Long Trigger
                            fill_price = lvl
                            
                            # Execute Long
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
        
        # Determine price bounds for gene normalization
        self.min_price = data_train['low'].min()
        self.max_price = data_train['high'].max()
        
        # Population: [Count, Level_1_Norm, ..., Level_100_Norm]
        # Gene 0: Number of levels (normalized 0-1, mapped to 5-100)
        # Genes 1-100: Price levels (normalized 0-1)
        self.population = np.random.rand(pop_size, MAX_LEVELS + 1)
        
    def decode_chromosome(self, chrom):
        # Decode Count
        count_norm = chrom[0]
        count = int(MIN_LEVELS + count_norm * (MAX_LEVELS - MIN_LEVELS))
        
        # Decode Levels
        levels_norm = chrom[1:]
        # Extract active levels
        active_indices = np.argsort(levels_norm)[:count] # Pick 'count' levels ? Or just first 'count'? 
        # Better: Use the first 'count' genes after sorting them to ensure consistency?
        # Strategy: Take first 'count' from the gene array, map to price, then sort.
        
        raw_levels = levels_norm[:count]
        prices = self.min_price + raw_levels * (self.max_price - self.min_price)
        prices.sort()
        return prices

    def fitness(self, chrom):
        levels = self.decode_chromosome(chrom)
        equity, _ = self.backtester.run(levels)
        
        # Objective: Maximize Final Equity
        # Penalty for 0 trades (avoid flatlines)
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
            
            # Selection (Tournament)
            new_pop = np.zeros_like(self.population)
            # Elitism
            new_pop[0] = self.population[best_idx]
            
            for i in range(1, self.pop_size):
                # Tourney size 3
                candidates = np.random.choice(self.pop_size, 3)
                parent1_idx = candidates[np.argmax(scores[candidates])]
                candidates = np.random.choice(self.pop_size, 3)
                parent2_idx = candidates[np.argmax(scores[candidates])]
                
                parent1 = self.population[parent1_idx]
                parent2 = self.population[parent2_idx]
                
                # Crossover
                if np.random.rand() < CROSSOVER_RATE:
                    cross_point = np.random.randint(1, len(parent1))
                    child = np.concatenate((parent1[:cross_point], parent2[cross_point:]))
                else:
                    child = parent1.copy()
                
                # Mutation
                mutation_mask = np.random.rand(len(child)) < MUTATION_RATE
                random_genes = np.random.rand(len(child))
                child[mutation_mask] = random_genes[mutation_mask]
                
                new_pop[i] = child
            
            self.population = new_pop
            
        # Return best chromosome
        final_scores = [self.fitness(p) for p in self.population]
        best_chrom = self.population[np.argmax(final_scores)]
        return best_chrom

def main():
    # 1. Download Data
    print("Downloading data...")
    r = requests.get(DATA_URL)
    if r.status_code != 200:
        raise Exception("Failed to download data")
    
    df = pd.read_csv(io.StringIO(r.text))
    
    # Clean column names
    df.columns = [c.lower() for c in df.columns]
    
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
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(equity, label='Equity Curve (Test Data)')
    plt.title(f'Strategy Performance (Test Data)\nLevels: {len(best_levels)}')
    plt.xlabel('Candles')
    plt.ylabel('Capital ($)')
    plt.legend()
    plt.grid(True)
    plt.savefig('output/equity.png')
    
    # Stats
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
        <img src="equity.png" style="max-width: 100%;">
        <hr>
        <h2>Price Levels</h2>
        <p>{list(np.round(best_levels, 2))}</p>
    </body>
    </html>
    """
    
    with open("output/index.html", "w") as f:
        f.write(html_content)
        
    print(f"Results saved to output/. Serving on port {PORT}...")
    
    # 6. Serve
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
