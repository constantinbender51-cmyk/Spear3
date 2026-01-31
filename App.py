import pandas as pd
import numpy as np
import pygad
import matplotlib.pyplot as plt
import http.server
import socketserver
import os
import sys
from urllib.parse import urlparse, parse_qs

# 1. Fetch Data
DATA_URL = "https://ohlcendpoint.up.railway.app/data/btc1h.csv"

def fetch_data(url):
    try:
        df = pd.read_csv(url)
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

# 2. Split Data (90/10)
split_idx = int(len(df) * 0.90)
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
# [0-31]: Price Levels
# [32]: SL % (0.5 - 2.0)
# [33]: TP % (0.5 - 10.0)

def backtest(genes, data, fee_pct=0.0):
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
        
        # Check Exit
        if position != 0:
            exit_executed = False
            if position == 1: # Long
                stop_price = entry_price * (1 - sl_pct)
                target_price = entry_price * (1 + tp_pct)
                
                if d_low[i] <= stop_price:
                    capital = capital * (stop_price / entry_price)
                    exit_executed = True
                elif d_high[i] >= target_price:
                    capital = capital * (target_price / entry_price)
                    exit_executed = True
                else:
                    current_equity = capital * (d_close[i] / entry_price)
            
            elif position == -1: # Short
                stop_price = entry_price * (1 + sl_pct)
                target_price = entry_price * (1 - tp_pct)
                
                if d_high[i] >= stop_price:
                    capital = capital * (entry_price / stop_price)
                    exit_executed = True
                elif d_low[i] <= target_price:
                    capital = capital * (entry_price / target_price)
                    exit_executed = True
                else:
                    current_equity = capital * (entry_price / d_close[i])
            
            if exit_executed:
                capital = capital * (1 - fee_pct) # Apply fee on exit
                position = 0
                current_equity = capital

        # Check Entry
        if position == 0:
            candle_low = d_low[i]
            candle_high = d_high[i]
            candle_open = d_open[i]
            
            for lvl in levels:
                if candle_low <= lvl <= candle_high:
                    entry_signal = 0
                    if candle_open < lvl:
                        entry_signal = -1
                    elif candle_open > lvl:
                        entry_signal = 1
                    
                    if entry_signal != 0:
                        position = entry_signal
                        entry_price = lvl
                        capital = capital * (1 - fee_pct) # Apply fee on entry
                        break
        
        equity_curve.append(current_equity if position != 0 else capital)

    return capital, equity_curve

# 4. GA Setup
price_min = np.min(train_data['low'])
price_max = np.max(train_data['high'])

gene_space = []
for _ in range(32):
    gene_space.append({'low': price_min, 'high': price_max})
gene_space.append({'low': 0.5, 'high': 2.0})
gene_space.append({'low': 0.5, 'high': 10.0})

def fitness_func(ga_instance, solution, solution_idx):
    # Optimize with 0 fee to find best raw strategy
    final_val, _ = backtest(solution, train_data, fee_pct=0.0)
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

solution, solution_fitness, solution_idx = ga_instance.best_solution()
print(f"Best Solution Parameters: {solution}")
print(f"Training Fitness (Equity): {solution_fitness}")

# 5. Serve Results
class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed_path = urlparse(self.path)
        if parsed_path.path == '/':
            # Parse Fee
            params = parse_qs(parsed_path.query)
            try:
                # Fee is passed as percentage (e.g. 0.5 for 0.5%)
                fee_display = float(params.get('fee', [0.0])[0])
                fee_pct = fee_display / 100.0
            except ValueError:
                fee_display = 0.0
                fee_pct = 0.0

            # Run Backtest with requested fee
            test_final_val, test_equity = backtest(solution, test_data, fee_pct=fee_pct)
            
            # Generate Plot
            plot_filename = "analysis_chart.png"
            plt.figure(figsize=(15, 12))
            
            # Subplot 1: Price Action
            ax1 = plt.subplot(2, 1, 1)
            ax1.plot(test_data['close'], label='Test Close Price', color='black', linewidth=1)
            optimized_levels = solution[:32]
            for lvl in optimized_levels:
                ax1.axhline(y=lvl, color='red', linestyle='--', alpha=0.3, linewidth=0.8)
            ax1.set_title('Test Data: Price Action with Optimized Reversal Levels')
            ax1.set_ylabel('Price')
            ax1.legend(loc='upper right')
            ax1.grid(True, alpha=0.3)
            
            # Subplot 2: Equity Curve
            ax2 = plt.subplot(2, 1, 2)
            ax2.plot(test_equity, label='Equity', color='blue', linewidth=1.5)
            ax2.set_title(f'Strategy Performance (Fee: {fee_display}%)\nFinal Equity: {test_final_val:.2f}')
            ax2.set_xlabel('Hours')
            ax2.set_ylabel('Equity ($)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(plot_filename)
            plt.close()
            
            # Serve HTML
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            
            html = f"""
            <html>
            <head>
                <title>Bot Results</title>
                <style>
                    body {{ font-family: sans-serif; padding: 20px; color: #333; }}
                    .container {{ max-width: 1200px; margin: 0 auto; }}
                    .controls {{ background: #f4f4f4; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
                    input[type=range] {{ width: 300px; }}
                </style>
                <script>
                    function updateFee(val) {{
                        document.getElementById('feeVal').innerText = val + '%';
                    }}
                    function applyFee(val) {{
                        window.location.href = '/?fee=' + val;
                    }}
                </script>
            </head>
            <body>
                <div class="container">
                    <h1>Optimization Results (90/10 Split)</h1>
                    
                    <div class="controls">
                        <label><b>Trading Fee:</b> <span id="feeVal">{fee_display}%</span></label><br>
                        <input type="range" min="0" max="1" step="0.01" value="{fee_display}" 
                               oninput="updateFee(this.value)" 
                               onchange="applyFee(this.value)">
                        <p><i>Adjust slider to recalculate test performance with fees.</i></p>
                    </div>

                    <div style="display: flex; justify-content: space-between;">
                        <div>
                            <h3>Parameters</h3>
                            <p><b>Optimized SL:</b> {solution[32]:.2f}%</p>
                            <p><b>Optimized TP:</b> {solution[33]:.2f}%</p>
                        </div>
                        <div>
                            <h3>Performance</h3>
                            <p><b>Training Fitness:</b> {solution_fitness:.2f}</p>
                            <p><b>Test Equity (Fee {fee_display}%):</b> {test_final_val:.2f}</p>
                        </div>
                    </div>

                    <hr>
                    <h2>Visual Analysis</h2>
                    <img src="/{plot_filename}" style="max-width: 100%; height: auto; border: 1px solid #ccc;" />
                </div>
            </body>
            </html>
            """
            self.wfile.write(html.encode())
        
        # Serve Plot Image
        elif parsed_path.path == '/analysis_chart.png':
            if os.path.exists('analysis_chart.png'):
                self.send_response(200)
                self.send_header("Content-type", "image/png")
                self.end_headers()
                with open('analysis_chart.png', 'rb') as f:
                    self.wfile.write(f.read())
            else:
                self.send_error(404)
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
