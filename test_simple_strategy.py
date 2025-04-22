import os
import pandas as pd
from bot.simple_strategy import SimpleStrategy

def test_different_balances():
    """Test the simple strategy with different initial balances"""
    # Load test data
    symbol = 'btc'
    year = '2023'
    file_path = f"data/{symbol}_{year}.csv"
    
    if not os.path.exists(file_path):
        print(f"Error: Test data not found at {file_path}")
        return
    
    # Load data
    df = pd.read_csv(file_path, index_col='timestamp', parse_dates=True)
    
    # Test with different initial balances
    test_balances = [100, 1000, 10000, 100000]
    
    results = []
    
    for balance in test_balances:
        # Create settings with fixed position size proportional to balance
        settings = {
            'short_window': 20,
            'long_window': 50,
            'initial_balance': balance,
            'fixed_position_size': 100  # Fixed amount regardless of balance
        }
        
        # Run backtest
        strategy = SimpleStrategy(df, settings)
        trades, final_balance = strategy.run_backtest()
        performance = strategy.get_performance()
        
        # Add to results
        results.append({
            'initial_balance': balance,
            'final_balance': performance['final_balance'],
            'profit': performance['profit'],
            'profit_percent': performance['profit_percent'],
            'total_trades': performance['total_trades'],
            'success_trades': performance['success_trades'],
            'failed_trades': performance['failed_trades'],
            'win_rate': performance['win_rate']
        })
    
    # Print results
    print("\nTest Results with Different Initial Balances:")
    print("-" * 80)
    print(f"{'Initial Balance':<15} {'Final Balance':<15} {'Profit':<15} {'Profit %':<10} {'Trades':<8} {'Win/Loss':<10} {'Win Rate':<10}")
    print("-" * 80)
    
    for result in results:
        print(f"{result['initial_balance']:<15.2f} {result['final_balance']:<15.2f} {result['profit']:<15.2f} {result['profit_percent']:<10.2f}% {result['total_trades']:<8} {result['success_trades']}/{result['failed_trades']:<6} {result['win_rate']:<10.2f}%")

if __name__ == "__main__":
    test_different_balances()
