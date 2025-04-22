import os
import sys
import pandas as pd
import json
from datetime import datetime

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from bot.advanced_strategy import AdvancedTradingBot

def backtest_for_year(symbol, year, settings=None):
    file_path = f"data/{symbol.lower()}_{year}.csv"
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return None

    df = pd.read_csv(file_path, index_col='timestamp', parse_dates=True)
    bot = AdvancedTradingBot(df, settings)
    trades, final_balance = bot.run_backtest()
    result = bot.get_performance()
    
    # Add metadata to result
    result['symbol'] = symbol.upper()
    result['year'] = year
    result['backtest_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Save trade log
    trades_df = pd.DataFrame(trades, columns=['Date', 'Action', 'Price', 'Amount'])
    trades_df.to_csv(f"data/{symbol.lower()}_{year}_trades.csv", index=False)
    
    # Save detailed performance metrics
    with open(f"data/{symbol.lower()}_{year}_performance.json", 'w') as f:
        json.dump(result, f, indent=4)
    
    print(f"✅ Backtested {symbol.upper()} for {year}:")
    print(f"   Final Balance = ${result['final_balance']:.2f}")
    print(f"   Profit = ${result['profit']:.2f} ({result['profit_percent']:.2f}%)")
    print(f"   Win Rate = {result['win_rate']:.2f}%")
    print(f"   Max Drawdown = {result['max_drawdown']:.2f}%")
    print(f"   Sharpe Ratio = {result['sharpe_ratio']:.2f}")
    
    return result

def optimize_parameters(symbol, year, param_grid):
    """
    Optimize strategy parameters using grid search
    param_grid: Dictionary of parameters to test with lists of values
    Example: {'sma_short': [10, 20, 30], 'sma_long': [50, 100]}
    """
    best_profit = -float('inf')
    best_params = None
    results = []
    
    # Generate all parameter combinations
    from itertools import product
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    
    for values in product(*param_values):
        params = dict(zip(param_names, values))
        print(f"Testing parameters: {params}")
        
        result = backtest_for_year(symbol, year, params)
        if result:
            results.append({**params, **result})
            
            if result['profit'] > best_profit:
                best_profit = result['profit']
                best_params = params
    
    # Save optimization results
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"data/{symbol.lower()}_{year}_optimization.csv", index=False)
    
    print("\n🔍 Optimization Results:")
    print(f"Best parameters: {best_params}")
    print(f"Best profit: ${best_profit:.2f}")
    
    return best_params

def run_all_backtests(settings=None):
    results = []
    for symbol in ['btc', 'eth']:
        for year in ['2020', '2021', '2022', '2023']:
            result = backtest_for_year(symbol, year, settings)
            if result:
                results.append(result)

    # Save summary report
    results_df = pd.DataFrame(results)
    results_df.to_csv('data/backtest_summary.csv', index=False)
    print("\n📊 Backtest Summary Saved to data/backtest_summary.csv")
    
    # Calculate aggregate statistics
    total_profit = sum(result['profit'] for result in results)
    avg_win_rate = sum(result['win_rate'] for result in results) / len(results)
    avg_drawdown = sum(result['max_drawdown'] for result in results) / len(results)
    
    print("\n📈 Aggregate Performance:")
    print(f"Total Profit: ${total_profit:.2f}")
    print(f"Average Win Rate: {avg_win_rate:.2f}%")
    print(f"Average Max Drawdown: {avg_drawdown:.2f}%")

if __name__ == "__main__":
    # Example of parameter optimization
    # param_grid = {
    #     'sma_short': [10, 15, 20, 25],
    #     'sma_long': [40, 50, 60],
    #     'rsi_period': [7, 14, 21],
    #     'position_size': [0.5, 0.75, 1.0]
    # }
    # best_params = optimize_parameters('btc', '2020', param_grid)
    
    # Run all backtests with default settings
    run_all_backtests()
