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
        # Ensure standard column names, normalize to lower case
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
split_idx = int(len(df) * 0.9)
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
# [0-31]: Price Levels (Reversal points)
# [32]: SL % (0.5 - 2.0)
# [33]: TP % (0.5 - 10.0)

def backtest(genes, data):
    levels = genes[:32]
    sl_pct = genes[32] / 100.0
    tp_pct = genes[33] / 100.0
    
    capital = 10000.0
    position = 0 # 0: None, 1: Long, -1: Short
    entry_price = 0.0
    
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
                
                if d_low[i] <= stop_price:
                    capital = capital * (stop_price / entry_price)
                    position = 0
                elif d_high[i] >= target_price:
                    capital = capital * (target_price / entry_price)
                    position = 0
                else:
                    current_equity = capital * (d_close[i] / entry_price)
            
            elif position == -1: # Short
                stop_price = entry_price * (1 + sl_pct)
                target_price = entry_price * (1 - tp_pct)
                
                if d_high[i] >= stop_price:
                    capital = capital * (entry_price / stop_price)
                    position = 0
                elif d_low[i] <= target_price:
                    capital = capital * (entry_price / target_price)
                    position = 0
                else:
                    current_equity = capital * (entry_price / d_close[i])

        # Check Entry if flat
        if position == 0:
            candle_low = d_low[i]
            candle_high = d_high[i]
            candle_open = d_open[i]
            
            for lvl in levels:
                if candle_low <= lvl <= candle_high:
                    if candle_open < lvl:
                        position = -1
                        entry_price = lvl
                        break
                    elif candle_open > lvl:
                        position = 1
                        entry_price = lvl
                        break
        
        equity_curve.append(current_equity if position != 0 else capital)

    return capital, equity_curve

# 4. GA Setup
price_min = np.min(train_data['low'])
price_max = np.max(train_data['high'])

gene_space = []
for _ in range(32):
    gene_space.append({'low': price_min, 'high': price_max}) # Levels
gene_space.append({'low': 0.5, 'high': 2.0}) # SL
gene_space.append({'low': 0.5, 'high': 10.0}) # TP

def fitness_func(ga_instance, solution, solution_idx):
    final_val, _ = backtest(solution, train_data)
    return final_val

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

test_final_val, test_equity = backtest(solution, test_data)
print(f"Test Set Final Equity: {test_final_val}")

# 6. Plotting
plot_filename = "analysis_chart.png"
plt.figure(figsize=(15, 12))

# Subplot 1: Price Action and Reversal Levels
ax1 = plt.subplot(2, 1, 1)
ax1.plot(test_data['close'], label='Test Close Price', color='black', linewidth=1)
optimized_levels = solution[:32]
# Plot levels as horizontal lines across the test period
for lvl in optimized_levels:
    ax1.axhline(y=lvl, color='red', linestyle='--', alpha=0.3, linewidth=0.8)
ax1.set_title('Test Data: Price Action with Optimized Reversal Levels')
ax1.set_ylabel('Price')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# Subplot 2: Equity Curve
ax2 = plt.subplot(2, 1, 2)
ax2.plot(test_equity, label='Equity', color='blue', linewidth=1.5)
ax2.set_title(f'Strategy Performance on Test Data\nFinal Equity: {test_final_val:.2f}')
ax2.set_xlabel('Hours')
ax2.set_ylabel('Equity ($)')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
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
            <head>
                <title>Bot Results</title>
                <style>body {{ font-family: sans-serif; padding: 20px; }}</style>
            </head>
            <body>
                <h1>Optimization Results</h1>
                <p><b>Training Equity:</b> {solution_fitness:.2f}</p>
                <p><b>Test Equity:</b> {test_final_val:.2f}</p>
                <p><b>Optimized SL:</b> {solution[32]:.2f}%</p>
                <p><b>Optimized TP:</b> {solution[33]:.2f}%</p>
                <hr>
                <h2>Visual Analysis</h2>
                <img src="/{plot_filename}" style="max-width: 100%; height: auto; border: 1px solid #ccc;" />
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
