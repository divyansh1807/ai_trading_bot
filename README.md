# AI Crypto Trading Bot

An advanced cryptocurrency trading bot with GUI for a college project. The bot implements simple trading strategies, tests them on historical crypto data (2020-2025), and displays detailed trade performance metrics.

## Features

- **Simple Trading Strategy**:
  - Simple Moving Average (SMA) Crossover
  - Fixed position sizing
  - Detailed trade tracking

- **Backtesting Engine**:
  - Test strategies on historical data (2020-2025)
  - Performance metrics and visualization
  - Configurable initial balance (default: $10,000)

- **Data Management**:
  - Historical data loading
  - Real-time data fetching
  - Simulated future data generation

- **Advanced GUI**:
  - Interactive controls
  - Performance graphs
  - Detailed trade summary with tabular view
  - Scrollable interface for better data visibility

## Project Structure

```
ai_crypto_bot/
├── backtest/                  # Backtesting modules
│   ├── __init__.py
│   ├── advanced_backtest.py   # Advanced backtesting engine
│   └── backtest_engine.py     # Basic backtesting engine
├── bot/                       # Trading bot modules
│   ├── __init__.py
│   ├── advanced_strategy.py   # Advanced trading strategy
│   ├── integrated_bot.py      # Integrated bot with all features
│   ├── ml_strategy.py         # Machine learning strategy
│   ├── sentiment_analysis.py  # Sentiment analysis module
│   ├── strategy.py            # Basic trading strategy
│   └── simple_strategy.py     # Simple SMA crossover strategy
├── data/                      # Data storage
│   └── backtest_summary.csv   # Summary of backtest results
├── gui/                       # GUI components
│   └── gui.py                 # Main GUI application
├── models/                    # ML model storage
├── results/                   # Backtest results
├── utils/                     # Utility modules
│   └── data_loader.py         # Data loading utilities
└── run_integrated_bot.py      # Main script to run the bot
```

## Getting Started

### Prerequisites

- Python 3.8+
- Required Python packages (install via `pip install -r requirements.txt`):
  - pandas
  - numpy
  - matplotlib
  - scikit-learn
  - requests
  - textblob
  - joblib

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/divyansh1807/ai_trading_bot.git
   cd ai_crypto_bot
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

### Usage

#### Running the GUI

```
python main.py
```

The GUI provides an intuitive interface for:
- Selecting cryptocurrency and year for backtesting
- Setting initial balance (default: $10,000)
- Viewing performance graphs
- Analyzing detailed trade summaries
- Scrolling through data for better visibility

#### Running from Command Line

```
python run_integrated_bot.py --mode backtest --symbol btc --year 2023
```

To run backtests for all symbols and years:
```
python run_integrated_bot.py --mode backtest --all
```

#### Running Live Trading (Simulation)

```
python run_integrated_bot.py --mode live --symbol btc
```


