import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import json

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from bot.advanced_strategy import AdvancedTradingBot

# Import optional components with error handling
try:
    from bot.ml_strategy import MLTradingStrategy
    ML_AVAILABLE = True
except ImportError:
    print("Warning: ML functionality not available. Install scikit-learn for ML features.")
    ML_AVAILABLE = False

try:
    from bot.sentiment_analysis import SentimentAnalyzer
    SENTIMENT_AVAILABLE = True
except ImportError:
    print("Warning: Sentiment analysis not available. Install textblob for sentiment features.")
    SENTIMENT_AVAILABLE = False

# Blockchain functionality has been removed
BLOCKCHAIN_AVAILABLE = False

class IntegratedTradingBot:
    def __init__(self, df, settings=None, use_ml=True, use_sentiment=True):
        """
        Initialize the integrated trading bot

        Args:
            df (DataFrame): Historical price data
            settings (dict): Trading bot settings
            use_ml (bool): Whether to use machine learning predictions
            use_sentiment (bool): Whether to use sentiment analysis

        """
        self.df = df.copy()

        # Default settings
        self.settings = {
            'sma_short': 20,
            'sma_long': 50,
            'rsi_period': 14,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'initial_balance': 10000,  # Default initial balance
            'position_size': 1.0,  # Percentage of balance to use (1.0 = 100%)
            'stop_loss': 0.05,     # 5% stop loss
            'ml_weight': 0.3,      # Weight for ML signals (0.0 to 1.0)
            'sentiment_weight': 0.2,  # Weight for sentiment signals (0.0 to 1.0)
            'technical_weight': 0.5  # Weight for technical signals (0.0 to 1.0)
        }

        # Update with custom settings if provided
        if settings:
            self.settings.update(settings)

        # Ensure initial_balance is always set to 10000 if not specified or invalid
        if 'initial_balance' not in self.settings or self.settings['initial_balance'] <= 0:
            self.settings['initial_balance'] = 10000

        # Initialize components
        self.technical_bot = AdvancedTradingBot(df, self.settings)

        # Initialize ML strategy if requested and available
        self.use_ml = use_ml and ML_AVAILABLE
        self.ml_strategy = None
        if self.use_ml:
            try:
                self.ml_strategy = MLTradingStrategy(model_type='random_forest')
                print("✅ ML strategy initialized")
            except Exception as e:
                print(f"❌ Failed to initialize ML strategy: {str(e)}")
                self.use_ml = False
        elif use_ml and not ML_AVAILABLE:
            print("❌ ML functionality not available. Install scikit-learn for ML features.")

        # Initialize sentiment analyzer if requested and available
        self.use_sentiment = use_sentiment and SENTIMENT_AVAILABLE
        self.sentiment_analyzer = None
        if self.use_sentiment:
            try:
                self.sentiment_analyzer = SentimentAnalyzer()
                print("✅ Sentiment analyzer initialized")
            except Exception as e:
                print(f"❌ Failed to initialize sentiment analyzer: {str(e)}")
                self.use_sentiment = False
        elif use_sentiment and not SENTIMENT_AVAILABLE:
            print("❌ Sentiment analysis not available. Install textblob for sentiment features.")

        # Blockchain functionality has been removed
        self.use_blockchain = False

        # Trading state
        self.balance = self.settings['initial_balance']
        self.position = 0  # Number of coins
        self.trades = []
        self.current_stop_loss = 0

    def prepare_data(self):
        """Prepare data for trading by calculating indicators and predictions"""
        # Calculate technical indicators
        self.technical_bot.calculate_indicators()

        # Get the DataFrame with indicators
        self.df = self.technical_bot.df.copy()

        # Add ML predictions if enabled
        if self.use_ml and self.ml_strategy:
            try:
                # Train the model if not already trained
                if not self.ml_strategy.model:
                    print("🔄 Training ML model...")
                    self.ml_strategy.train(self.df)

                # Get predictions
                ml_predictions = self.ml_strategy.predict(self.df)

                # Add predictions to the DataFrame
                self.df['ml_prediction'] = ml_predictions['ml_prediction']
                self.df['ml_signal'] = ml_predictions['ml_signal']

                print("✅ ML predictions added")
            except Exception as e:
                print(f"❌ Failed to add ML predictions: {str(e)}")
                self.use_ml = False

        # Add sentiment analysis if enabled
        if self.use_sentiment and self.sentiment_analyzer:
            try:
                # Get the symbol from the DataFrame name or a default
                symbol = getattr(self.df, 'name', 'BTC')

                # Get sentiment
                sentiment = self.sentiment_analyzer.get_combined_sentiment(symbol)

                if sentiment:
                    # Add sentiment to the DataFrame (same value for all rows)
                    self.df['sentiment_score'] = sentiment['sentiment_score']
                    self.df['sentiment_signal'] = sentiment['signal']

                    print(f"✅ Sentiment analysis added: {sentiment['overall_sentiment']}")
                else:
                    print("❌ No sentiment data available")
                    self.use_sentiment = False
            except Exception as e:
                print(f"❌ Failed to add sentiment analysis: {str(e)}")
                self.use_sentiment = False

    def generate_combined_signals(self):
        """Generate combined trading signals using all enabled strategies"""
        # Ensure data is prepared
        if 'combined_signal' not in self.df.columns:
            self.prepare_data()

        # Get weights
        technical_weight = self.settings['technical_weight']
        ml_weight = self.settings['ml_weight'] if self.use_ml else 0
        sentiment_weight = self.settings['sentiment_weight'] if self.use_sentiment else 0

        # Normalize weights
        total_weight = technical_weight + ml_weight + sentiment_weight
        if total_weight > 0:
            technical_weight /= total_weight
            ml_weight /= total_weight
            sentiment_weight /= total_weight

        # Initialize combined signal column
        self.df['integrated_signal'] = 0

        # Add technical signals
        self.df['integrated_signal'] += self.df['combined_signal'] * technical_weight

        # Add ML signals if enabled
        if self.use_ml and 'ml_signal' in self.df.columns:
            self.df['integrated_signal'] += self.df['ml_signal'] * ml_weight

        # Add sentiment signals if enabled
        if self.use_sentiment and 'sentiment_signal' in self.df.columns:
            self.df['integrated_signal'] += self.df['sentiment_signal'] * sentiment_weight

        # Generate positions (1 for buy, -1 for sell, 0 for hold)
        self.df['integrated_positions'] = 0
        self.df.loc[self.df['integrated_signal'] >= 0.5, 'integrated_positions'] = 1
        self.df.loc[self.df['integrated_signal'] <= -0.5, 'integrated_positions'] = -1

        # Convert to actual trade signals (entry/exit points)
        self.df['integrated_trade_signal'] = self.df['integrated_positions'].diff()

    def run_backtest(self):
        """Run backtest using the integrated strategy"""
        # Generate signals
        self.generate_combined_signals()

        # Skip the initial period where indicators are being calculated
        start_index = max(
            self.settings['sma_long'],
            self.settings['rsi_period'],
            26  # For MACD
        )

        for i in range(start_index, len(self.df)):
            date = self.df.index[i]
            row = self.df.iloc[i]
            price = row['close']
            signal = row['integrated_trade_signal']

            # Check stop loss if we have a position
            if self.position > 0 and price <= self.current_stop_loss:
                # Stop loss triggered
                self.balance = self.position * price
                self.trades.append((date, 'STOP_LOSS', price, self.balance))
                self.position = 0
                continue

            if signal == 1:  # BUY signal
                if self.balance > 0:
                    # Calculate position size based on settings
                    amount_to_invest = self.balance * self.settings['position_size']
                    self.position = amount_to_invest / price
                    self.trades.append((date, 'BUY', price, amount_to_invest))
                    self.balance -= amount_to_invest

                    # Set stop loss
                    self.current_stop_loss = price * (1 - self.settings['stop_loss'])

            elif signal == -1:  # SELL signal
                if self.position > 0:
                    sell_value = self.position * price
                    self.balance += sell_value
                    self.trades.append((date, 'SELL', price, sell_value))
                    self.position = 0
                    self.current_stop_loss = 0

        # Final balance (if holding position at end)
        if self.position > 0:
            final_price = self.df.iloc[-1]['close']
            final_value = self.position * final_price
            self.balance += final_value
            self.trades.append((self.df.index[-1], 'FINAL_SELL', final_price, final_value))
            self.position = 0

        return self.trades, self.balance

    def get_performance(self):
        """Get performance metrics"""
        if not self.trades:
            return {
                'final_balance': self.settings['initial_balance'],
                'profit': 0,
                'profit_percent': 0,
                'success_trades': 0,
                'failed_trades': 0,
                'total_trades': 0,
                'win_rate': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'strategies_used': {
                    'technical': True,
                    'ml': self.use_ml,
                    'sentiment': self.use_sentiment
                }
            }

        initial_balance = self.settings['initial_balance']
        final_balance = self.balance
        profit = final_balance - initial_balance
        profit_percent = (profit / initial_balance) * 100

        # Calculate trade statistics
        buy_trades = [t for t in self.trades if t[1] == 'BUY']
        sell_trades = [t for t in self.trades if t[1] in ('SELL', 'FINAL_SELL', 'STOP_LOSS')]

        # Pair buy and sell trades to determine winners and losers
        trade_pairs = []
        buy_index = 0

        for i in range(len(sell_trades)):
            if buy_index < len(buy_trades):
                buy_trade = buy_trades[buy_index]
                sell_trade = sell_trades[i]

                buy_price = buy_trade[2]
                sell_price = sell_trade[2]

                trade_pairs.append({
                    'buy_date': buy_trade[0],
                    'buy_price': buy_price,
                    'sell_date': sell_trade[0],
                    'sell_price': sell_price,
                    'profit': (sell_price - buy_price) / buy_price * 100
                })

                buy_index += 1

        # Calculate win rate
        winning_trades = [t for t in trade_pairs if t['profit'] > 0]
        success_trades = len(winning_trades)
        failed_trades = len(trade_pairs) - success_trades
        win_rate = success_trades / len(trade_pairs) if trade_pairs else 0

        # Calculate maximum drawdown
        balance_history = [initial_balance]
        for trade in self.trades:
            if trade[1] == 'BUY':
                balance_history.append(balance_history[-1] - trade[3])
            else:
                balance_history.append(balance_history[-1] + trade[3])

        peak = initial_balance
        max_drawdown = 0

        for balance in balance_history:
            if balance > peak:
                peak = balance
            drawdown = (peak - balance) / peak
            max_drawdown = max(max_drawdown, drawdown)

        # Calculate Sharpe ratio (simplified)
        if trade_pairs:
            returns = [t['profit'] / 100 for t in trade_pairs]
            mean_return = sum(returns) / len(returns)
            std_return = np.std(returns) if len(returns) > 1 else 0
            sharpe_ratio = mean_return / std_return if std_return > 0 else 0
        else:
            sharpe_ratio = 0

        return {
            'final_balance': final_balance,
            'profit': profit,
            'profit_percent': profit_percent,
            'success_trades': success_trades,
            'failed_trades': failed_trades,
            'total_trades': len(trade_pairs),
            'win_rate': win_rate * 100,  # as percentage
            'max_drawdown': max_drawdown * 100,  # as percentage
            'sharpe_ratio': sharpe_ratio,
            'strategies_used': {
                'technical': True,
                'ml': self.use_ml,
                'sentiment': self.use_sentiment
            }
        }

    def save_results(self, symbol, year):
        """Save backtest results to files"""
        # Create results directory
        os.makedirs("results", exist_ok=True)

        # Save trades
        trades_df = pd.DataFrame(self.trades, columns=['Date', 'Action', 'Price', 'Amount'])
        trades_df.to_csv(f"results/{symbol.lower()}_{year}_integrated_trades.csv", index=False)

        # Save performance metrics
        performance = self.get_performance()

        # Add metadata
        performance['symbol'] = symbol.upper()
        performance['year'] = year
        performance['backtest_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Save as JSON
        with open(f"results/{symbol.lower()}_{year}_integrated_performance.json", 'w') as f:
            json.dump(performance, f, indent=4)

        print(f"✅ Results saved for {symbol.upper()} {year}")

        return performance


