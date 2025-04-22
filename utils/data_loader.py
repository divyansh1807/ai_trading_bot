import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import os

def get_binance_ohlcv(symbol, start_date, end_date, interval='1d'):
    base_url = "https://api.binance.com/api/v3/klines"

    # Handle both string dates and datetime objects
    if isinstance(start_date, str):
        start = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
    else:
        start = int(start_date.timestamp() * 1000)

    if isinstance(end_date, str):
        end = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)
    else:
        end = int(end_date.timestamp() * 1000)

    all_data = []

    while start < end:
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': 1000,
            'startTime': start
        }

        response = requests.get(base_url, params=params)
        data = response.json()

        if not data:
            break

        all_data.extend(data)

        last_time = data[-1][0]
        start = last_time + 1
        time.sleep(0.5)

    df = pd.DataFrame(all_data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_volume', 'taker_buy_quote_volume', 'ignore'
    ])

    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    return df

def save_yearly_data(symbol='BTCUSDT', include_current_year=True, include_future=False):
    os.makedirs("data", exist_ok=True)

    # Base years
    years = ['2020', '2021', '2022', '2023']

    # Get current year
    current_year = datetime.now().year

    # Make sure 2024 is included
    if '2024' not in years:
        years.append('2024')

    # Add current year if requested and not already in the list
    if include_current_year and str(current_year) not in years:
        years.append(str(current_year))

    # Add future year for simulation if requested
    if include_future and str(current_year + 1) not in years:
        years.append(str(current_year + 1))

    for year in years:
        year_int = int(year)
        current_year_int = datetime.now().year

        # For historical and current year, use actual data
        if year_int <= current_year_int:
            start = f"{year}-01-01"

            # For current year, end date is today
            if year_int == current_year_int:
                end = datetime.now().strftime('%Y-%m-%d')
            else:
                end = f"{year}-12-31"

            print(f"Downloading {symbol} data for {year}...")

            try:
                df = get_binance_ohlcv(symbol, start, end)
                ticker = 'btc' if symbol == 'BTCUSDT' else 'eth'
                df.to_csv(f"data/{ticker}_{year}.csv")
                print(f"Saved: data/{ticker}_{year}.csv")
            except Exception as e:
                print(f"Error downloading {symbol} data for {year}: {str(e)}")

        # For future year, generate simulated data
        else:
            print(f"Generating simulated data for {symbol} for {year}...")

            # Use the last year's data as a base and add some random variations
            ticker = 'btc' if symbol == 'BTCUSDT' else 'eth'
            last_year = str(year_int - 1)

            try:
                # Load last year's data
                last_year_file = f"data/{ticker}_{last_year}.csv"
                if os.path.exists(last_year_file):
                    last_df = pd.read_csv(last_year_file, index_col='timestamp', parse_dates=True)

                    # Create a date range for the future year
                    start_date = datetime(year_int, 1, 1)
                    end_date = datetime(year_int, 12, 31)
                    future_dates = pd.date_range(start=start_date, end=end_date, freq='D')

                    # Create a new DataFrame with the future dates
                    future_df = pd.DataFrame(index=future_dates)
                    future_df.index.name = 'timestamp'

                    # Get the last price from the previous year
                    last_price = float(last_df['close'].iloc[-1])

                    # Generate simulated prices with random walk and some seasonality
                    import numpy as np
                    np.random.seed(42)  # For reproducibility

                    # Parameters for simulation
                    volatility = last_df['close'].pct_change().std() * np.sqrt(252)  # Annualized volatility
                    drift = last_df['close'].pct_change().mean() * 252  # Annualized drift

                    # Generate daily returns
                    daily_returns = np.random.normal(drift/252, volatility/np.sqrt(252), len(future_dates))

                    # Add some seasonality
                    seasonality = 0.1 * np.sin(np.linspace(0, 2*np.pi, len(future_dates)))
                    daily_returns = daily_returns + seasonality

                    # Calculate prices
                    prices = [last_price]
                    for ret in daily_returns:
                        prices.append(prices[-1] * (1 + ret))
                    prices = prices[1:]  # Remove the initial price

                    # Create OHLC data
                    future_df['close'] = prices
                    future_df['open'] = future_df['close'].shift(1) * (1 + np.random.normal(0, 0.01, len(future_dates)))
                    future_df['high'] = future_df[['open', 'close']].max(axis=1) * (1 + abs(np.random.normal(0, 0.01, len(future_dates))))
                    future_df['low'] = future_df[['open', 'close']].min(axis=1) * (1 - abs(np.random.normal(0, 0.01, len(future_dates))))

                    # Generate volume data based on historical patterns
                    avg_volume = last_df['volume'].mean()
                    std_volume = last_df['volume'].std()
                    future_df['volume'] = np.random.normal(avg_volume, std_volume, len(future_dates))
                    future_df['volume'] = future_df['volume'].clip(lower=avg_volume*0.1)  # Ensure positive volume

                    # Fill NaN values
                    future_df = future_df.fillna(method='bfill')

                    # Save to CSV
                    future_df.to_csv(f"data/{ticker}_{year}.csv")
                    print(f"Saved simulated data: data/{ticker}_{year}.csv")
                else:
                    print(f"Cannot generate simulated data: {last_year_file} not found")
            except Exception as e:
                print(f"Error generating simulated data for {year}: {str(e)}")

def save_all_data(include_current_year=True, include_future=False):
    save_yearly_data('BTCUSDT', include_current_year, include_future)
    save_yearly_data('ETHUSDT', include_current_year, include_future)

def get_latest_data(symbol='BTCUSDT', days=30):
    """Get the latest data for a symbol

    Args:
        symbol (str): Trading pair symbol
        days (int): Number of days to look back

    Returns:
        DataFrame: Latest price data
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    start = start_date.strftime('%Y-%m-%d')
    end = end_date.strftime('%Y-%m-%d')

    print(f"Downloading latest {symbol} data from {start} to {end}...")

    try:
        df = get_binance_ohlcv(symbol, start, end)
        return df
    except Exception as e:
        print(f"Error downloading latest data: {str(e)}")
        return None

if __name__ == "__main__":
    # Download historical data and generate future data
    save_all_data(include_current_year=True, include_future=True)
