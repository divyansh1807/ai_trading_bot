import os
import sys
import pandas as pd

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from bot.strategy import TradingBot

def backtest_for_year(symbol, year):
    file_path = f"data/{symbol.lower()}_{year}.csv"
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return None

    df = pd.read_csv(file_path, index_col='timestamp', parse_dates=True)
    bot = TradingBot(df)
    trades, final_balance = bot.run_backtest()
    result = bot.get_performance()

    # Save trade log
    trades_df = pd.DataFrame(bot.trades, columns=['Date', 'Action', 'Price', 'Balance'])
    trades_df.to_csv(f"data/{symbol.lower()}_{year}_trades.csv", index=False)

    print(f"✅ Backtested {symbol.upper()} for {year}: Final Balance = ${result['final_balance']:.2f}, Profit = ${result['profit']:.2f}")
    return result

def run_all_backtests():
    results = []
    for symbol in ['btc', 'eth']:
        for year in ['2020', '2021', '2022', '2023']:
            result = backtest_for_year(symbol, year)
            if result:
                results.append({
                    'Symbol': symbol.upper(),
                    'Year': year,
                    **result
                })

    # Save summary report
    results_df = pd.DataFrame(results)
    results_df.to_csv('data/backtest_summary.csv', index=False)
    print("\n📊 Backtest Summary Saved to data/backtest_summary.csv")

if __name__ == "__main__":
    run_all_backtests()
