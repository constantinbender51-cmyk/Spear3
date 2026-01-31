import pandas as pd
import numpy as np
import pygad
import matplotlib.pyplot as plt
import http.server
import socketserver
import os
import sys

# 1. Fetch Data
DATA_URL = "https://ohlcendpoint.up.railway.app/data/btc1h.csv"

def fetch_data(url):
    try:
        df = pd.read_csv(url)
        # Ensure standard column names if necessary, assuming CSV has correct headers
        # or implies standard OHLC structure.
        # We assume columns: timestamp, open, high, low, close, volume (or similar)
        # Normalizing column names to lower case for consistency
        df.columns = [c.lower() for c in df.columns]
        return df
    except Exception as e:
        print(f"Error fetching data: {e}")
        sys.exit(1)

df = fetch_data(DATA_URL)
prices = df['close'].values
highs = df['high'].values
lows = df['low'].values
opens = df['open'].values

# 2. Split Data (70/30)
split_idx = int(len(df) * 0.70)
train_data = {
    'open': opens[:split_idx],
    'high': highs[:split_idx],
    'low': lows[:split_idx],
    'close': prices[:split_idx]
}
test_data = {
    'open': opens[split_idx:],
    'high': highs[split_idx:],
    'low': lows[split_idx:],
    'close': prices[split_idx:]
}

# 3. Define Strategy Logic
# Logic: 34 variables.
# [0-31]: Price Levels (Reversal points)
# [32]: SL % (0.5 - 2.0)
# [33]: TP % (0.5 - 10.0)

def backtest(genes, data):
    """
    Backtests the reversal strategy.
    
    Args:
        genes: Array of 34 floats.
        data: Dict of OHLC numpy arrays.
        
    Returns:
        Final Equity (assuming starting capital 10000)
    """
    levels = genes[:32]
    sl_pct = genes[32] / 100.0
    tp_pct = genes[33] / 100.0
    
    capital = 10000.0
    position = 0 # 0: None, 1: Long, -1: Short
    entry_price = 0.0
    
    # Extract arrays for speed
    d_open = data['open']
    d_high = data['high']
    d_low = data['low']
    d_close = data['close']
    n = len(d_close)
    
    equity_curve = [capital]
    
    for i in range(n):
        current_equity = capital
        
        # Check Exit if in position
        if position != 0:
            if position == 1: # Long
                stop_price = entry_price * (1 - sl_pct)
                target_price = entry_price * (1 + tp_pct)
                
                # Check Low for SL
                if d_low[i] <= stop_price:
                    capital = capital * (stop_price / entry_price)
                    position = 0
                # Check High for TP
                elif d_high[i] >= target_price:
                    capital = capital * (target_price / entry_price)
                    position = 0
                else:
                    # Mark to market (approximate for equity curve)
                    current_equity = capital * (d_close[i] / entry_price)
            
            elif position == -1: # Short
                stop_price = entry_price * (1 + sl_pct)
                target_price = entry_price * (1 - tp_pct)
                
                # Check High for SL
                if d_high[i] >= stop_price:
                    # Short loss: entry/exit
                    capital = capital * (entry_price / stop_price) # Inverse relation
                    position = 0
                # Check Low for TP
                elif d_low[i] <= target_price:
                    capital = capital * (entry_price / target_price)
                    position = 0
                else:
                    current_equity = capital * (entry_price / d_close[i])

        # Check Entry if flat
        # Reversal Logic:
        # If High >= Level >= Low
        #   If Open < Level: Hit from below -> Resistance -> Short
        #   If Open > Level: Hit from above -> Support -> Long
        if position == 0:
            # Vectorized check is hard with 32 levels, iterating levels
            # Optimization: Sort levels or use broadcasting? 
            # For 32 levels and O(N) loop, inner loop is 32 operations. Acceptable.
            
            candle_low = d_low[i]
            candle_high = d_high[i]
            candle_open = d_open[i]
            
            # Find levels intersected by this candle
            # Level must be between Low and High
            
            # Simple greedy approach: Take the first valid trigger
            for lvl in levels:
                if candle_low <= lvl <= candle_high:
                    if candle_open < lvl:
                        # Crossing Up -> Short
                        position = -1
                        entry_price = lvl
                        break
                    elif candle_open > lvl:
                        # Crossing Down -> Long
                        position = 1
                        entry_price = lvl
                        break
        
        equity_curve.append(current_equity if position != 0 else capital)

    return capital, equity_curve

# 4. GA Setup
# Gene space: 
# Levels: Min/Max of training data close
price_min = np.min(train_data['low'])
price_max = np.max(train_data['high'])

# Gene Space Definition
# 32 genes: price_min to price_max
# 1 gene: SL (0.5 to 2.0)
# 1 gene: TP (0.5 to 10.0)
gene_space = []
for _ in range(32):
    gene_space.append({'low': price_min, 'high': price_max}) # Levels
gene_space.append({'low': 0.5, 'high': 2.0}) # SL
gene_space.append({'low': 0.5, 'high': 10.0}) # TP

def fitness_func(ga_instance, solution, solution_idx):
    final_val, _ = backtest(solution, train_data)
    # Fitness is profit. If profit < 0 (loss), fitness is low.
    # PyGAD maximizes fitness.
    return final_val

# GA Parameters
# Keeping complexity high as requested
num_generations = 20
num_parents_mating = 4
sol_per_pop = 10
num_genes = 34

ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_func,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       gene_space=gene_space,
                       mutation_type="random",
                       mutation_percent_genes=10,
                       suppress_warnings=True)

print("Starting GA Optimization...")
ga_instance.run()

# 5. Test Results
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print(f"Best Solution Parameters: {solution}")
print(f"Training Fitness (Equity): {solution_fitness}")

# Run on Test Set
test_final_val, test_equity = backtest(solution, test_data)
print(f"Test Set Final Equity: {test_final_val}")

# 6. Plotting
plt.figure(figsize=(12, 6))
plt.plot(test_equity, label='Test Equity Curve')
plt.title(f'Strategy Performance on Test Data\nFinal Equity: {test_final_val:.2f}')
plt.xlabel('Hours')
plt.ylabel('Equity ($)')
plt.legend()
plt.grid(True)
plot_filename = "equity_curve.png"
plt.savefig(plot_filename)
plt.close()

# 7. Serve Results
class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            html = f"""
            <html>
            <head><title>Bot Results</title></head>
            <body>
                <h1>Optimization Results</h1>
                <p><b>Training Equity:</b> {solution_fitness:.2f}</p>
                <p><b>Test Equity:</b> {test_final_val:.2f}</p>
                <p><b>Optimized SL:</b> {solution[32]:.2f}%</p>
                <p><b>Optimized TP:</b> {solution[33]:.2f}%</p>
                <h2>Equity Curve</h2>
                <img src="/{plot_filename}" />
            </body>
            </html>
            """
            self.wfile.write(html.encode())
        else:
            super().do_GET()

PORT = 8080
with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print(f"Serving results at http://localhost:{PORT}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.server_close()
        print("Server stopped.")
