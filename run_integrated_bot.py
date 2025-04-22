import os
import sys
import pandas as pd
import numpy as np
import argparse
from datetime import datetime, timedelta

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import with error handling
try:
    from bot.simple_strategy import SimpleStrategy
    BOT_AVAILABLE = True
except ImportError as e:
    print(f"Error importing SimpleStrategy: {str(e)}")
    BOT_AVAILABLE = False

try:
    from utils.data_loader import save_yearly_data, get_latest_data
    DATA_LOADER_AVAILABLE = True
except ImportError as e:
    print(f"Error importing data loader: {str(e)}")
    DATA_LOADER_AVAILABLE = False

def create_simulated_data(symbol, year):
    """
    Create simulated price data for a future year

    Args:
        symbol (str): Cryptocurrency symbol (e.g., 'btc', 'eth')
        year (str): Year to simulate (e.g., '2024', '2025')
    """
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)

    # Set parameters for simulation
    year_int = int(year)
    start_date = datetime(year_int, 1, 1)
    end_date = datetime(year_int, 12, 31)
    days = (end_date - start_date).days + 1

    # Create date range
    date_range = [start_date + timedelta(days=i) for i in range(days)]

    # Set initial price based on symbol
    if symbol.lower() == 'btc':
        initial_price = 50000.0  # Example initial price for BTC
    else:  # ETH
        initial_price = 3000.0   # Example initial price for ETH

    # Generate random price movements
    np.random.seed(42)  # For reproducibility
    daily_returns = np.random.normal(0.001, 0.02, days)  # Mean daily return and volatility

    # Add some trend and seasonality
    trend = np.linspace(0, 0.5, days)  # Upward trend
    seasonality = 0.1 * np.sin(np.linspace(0, 4*np.pi, days))  # Seasonal pattern

    # Combine components
    adjusted_returns = daily_returns + 0.001 * trend + seasonality

    # Calculate prices
    prices = [initial_price]
    for ret in adjusted_returns:
        prices.append(prices[-1] * (1 + ret))

    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': date_range,
        'open': prices[:-1],
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices[:-1]],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices[:-1]],
        'close': prices[1:],
        'volume': [abs(np.random.normal(1000000, 500000)) for _ in range(days)]
    })

    # Set timestamp as index
    df.set_index('timestamp', inplace=True)

    # Save to CSV
    file_path = f"data/{symbol.lower()}_{year}.csv"
    df.to_csv(file_path)

    print(f"✅ Created simulated data: {file_path}")

def run_backtest(symbol, year, settings=None):
    # Ensure settings has a default value
    if settings is None:
        settings = {}

    # Always set initial_balance to 10000 if not specified
    if 'initial_balance' not in settings:
        settings['initial_balance'] = 10000

    """
    Run backtest for a specific symbol and year using the simple strategy

    Args:
        symbol (str): Cryptocurrency symbol (e.g., 'btc', 'eth')
        year (str): Year to backtest (e.g., '2020', '2021')
        settings (dict): Custom settings for the trading bot

    Returns:
        dict: Performance metrics
    """
    # Load data
    file_path = f"data/{symbol.lower()}_{year}.csv"
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        print("Downloading data...")
        try:
            # Create data directory if it doesn't exist
            os.makedirs("data", exist_ok=True)

            # Try to download data
            save_yearly_data(f"{symbol.upper()}USDT", include_current_year=True, include_future=True)
        except Exception as e:
            print(f"❌ Error downloading data: {str(e)}")

            # If it's a future year, create a simulated data file
            current_year = datetime.now().year
            if int(year) > current_year:
                print(f"🔮 Creating simulated data for {year}...")
                try:
                    # Create a simple simulated data file
                    create_simulated_data(symbol, year)
                except Exception as e:
                    print(f"❌ Error creating simulated data: {str(e)}")

    # Check again if file exists
    if not os.path.exists(file_path):
        print(f"❌ Failed to get data for {symbol.upper()} {year}")
        return None

    # Load data
    df = pd.read_csv(file_path, index_col='timestamp', parse_dates=True)

    # Initialize bot with simple strategy
    bot = SimpleStrategy(df, settings=settings)

    # Run backtest
    print(f"🔄 Running backtest for {symbol.upper()} {year}...")
    bot.run_backtest()

    # Get performance metrics
    performance = bot.get_performance()

    # Save results
    bot.save_results(symbol, year)

    # Print summary
    print(f"\n📊 Backtest Results for {symbol.upper()} {year}:")
    print(f"Final Balance: ${performance['final_balance']:.2f}")
    print(f"Profit: ${performance['profit']:.2f} ({performance['profit_percent']:.2f}%)")
    print(f"Win Rate: {performance['win_rate']:.2f}%")
    print(f"Max Drawdown: {performance['max_drawdown']:.2f}%")
    print(f"Sharpe Ratio: {performance['sharpe_ratio']:.2f}")
    print(f"Total Trades: {performance['total_trades']}")

    return performance

def run_all_backtests(settings=None):
    # Ensure settings has a default value
    if settings is None:
        settings = {}

    # Always set initial_balance to 10000 if not specified
    if 'initial_balance' not in settings:
        settings['initial_balance'] = 10000

    """
    Run backtests for all symbols and years using the simple strategy

    Args:
        settings (dict): Custom settings for the trading bot

    Returns:
        list: Performance metrics for all backtests
    """
    results = []

    for symbol in ['btc', 'eth']:
        for year in ['2020', '2021', '2022', '2023', '2024', '2025']:
            performance = run_backtest(
                symbol,
                year,
                settings=settings
            )

            if performance:
                results.append({
                    'symbol': symbol.upper(),
                    'year': year,
                    **performance
                })

    # Save summary report
    results_df = pd.DataFrame(results)
    results_df.to_csv('results/integrated_backtest_summary.csv', index=False)

    # Calculate aggregate statistics
    total_profit = sum(result['profit'] for result in results)
    avg_win_rate = sum(result['win_rate'] for result in results) / len(results) if results else 0
    avg_drawdown = sum(result['max_drawdown'] for result in results) / len(results) if results else 0

    print("\n📈 Aggregate Performance:")
    print(f"Total Profit: ${total_profit:.2f}")
    print(f"Average Win Rate: {avg_win_rate:.2f}%")
    print(f"Average Max Drawdown: {avg_drawdown:.2f}%")

    return results

def run_live_trading(symbol, settings=None):
    # Ensure settings has a default value
    if settings is None:
        settings = {}

    # Always set initial_balance to 10000 if not specified
    if 'initial_balance' not in settings:
        settings['initial_balance'] = 10000

    """
    Run live trading for a specific symbol using the simple strategy

    Args:
        symbol (str): Cryptocurrency symbol (e.g., 'btc', 'eth')
        settings (dict): Custom settings for the trading bot

    Returns:
        None
    """
    # Get latest data
    df = get_latest_data(f"{symbol.upper()}USDT", days=30)

    if df is None:
        print(f"❌ Failed to get latest data for {symbol.upper()}")
        return

    # Initialize bot with simple strategy
    bot = SimpleStrategy(df, settings=settings)

    # Generate signals and run a quick backtest
    bot.generate_signals()

    # Get the latest signal
    latest_signal = bot.df['positions'].diff().iloc[-1]

    # Display signal
    if latest_signal == 1:  # BUY signal
        print(f"🔄 BUY signal detected for {symbol.upper()}")
    elif latest_signal == -1:  # SELL signal
        print(f"🔄 SELL signal detected for {symbol.upper()}")
    else:
        print(f"ℹ️ No trading signal for {symbol.upper()}")

    print("Note: Live trading execution has been removed with blockchain functionality.")

if __name__ == "__main__":
    # Check if required components are available
    if not BOT_AVAILABLE:
        print("\n❌ Error: SimpleStrategy is not available. Cannot proceed.")
        sys.exit(1)

    if not DATA_LOADER_AVAILABLE:
        print("\n❌ Error: Data loader is not available. Cannot proceed.")
        sys.exit(1)

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run AI Crypto Trading Bot')

    parser.add_argument('--mode', choices=['backtest', 'live'], default='backtest',
                        help='Trading mode: backtest or live')

    parser.add_argument('--symbol', choices=['btc', 'eth'], default='btc',
                        help='Cryptocurrency symbol')

    parser.add_argument('--year', choices=['2020', '2021', '2022', '2023', '2024', '2025'], default='2023',
                        help='Year for backtesting')

    parser.add_argument('--all', action='store_true',
                        help='Run backtest for all symbols and years')

    # Remove ML and sentiment options as we're using a simple strategy



    args = parser.parse_args()

    # Custom settings for simple strategy
    settings = {
        'short_window': 20,
        'long_window': 50,
        'initial_balance': 10000,
        'fixed_position_size': 1000  # Fixed amount to invest per trade
    }

    try:
        # Run in selected mode
        if args.mode == 'backtest':
            if args.all:
                run_all_backtests(
                    settings=settings
                )
            else:
                run_backtest(
                    args.symbol,
                    args.year,
                    settings=settings
                )

        elif args.mode == 'live':
            run_live_trading(
                args.symbol,
                settings=settings
            )
    except Exception as e:
        print(f"\n❌ Error running the bot: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
