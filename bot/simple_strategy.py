import pandas as pd
import numpy as np

class SimpleStrategy:
    def __init__(self, df, settings=None):
        """
        Initialize the simple trading strategy
        
        Args:
            df (DataFrame): Historical price data
            settings (dict): Strategy settings
        """
        self.df = df.copy()
        
        # Default settings
        self.settings = {
            'short_window': 20,
            'long_window': 50,
            'initial_balance': 10000,
            'fixed_position_size': 1000  # Fixed amount to invest per trade
        }
        
        # Update with custom settings if provided
        if settings:
            self.settings.update(settings)
            
        # Ensure initial_balance is always set to 10000 if not specified or invalid
        if 'initial_balance' not in self.settings or self.settings['initial_balance'] <= 0:
            self.settings['initial_balance'] = 10000
            
        self.balance = self.settings['initial_balance']
        self.position = 0  # Number of coins
        self.trades = []
    
    def generate_signals(self):
        """Generate trading signals based on SMA crossover"""
        # Calculate simple moving averages
        self.df['SMA_short'] = self.df['close'].rolling(window=self.settings['short_window']).mean()
        self.df['SMA_long'] = self.df['close'].rolling(window=self.settings['long_window']).mean()
        
        # Generate signal (1 when short SMA > long SMA, 0 otherwise)
        self.df['signal'] = 0
        self.df.loc[self.df['SMA_short'] > self.df['SMA_long'], 'signal'] = 1
        
        # Generate trading positions (1 for buy, -1 for sell, based on signal changes)
        self.df['positions'] = self.df['signal'].diff()
    
    def run_backtest(self):
        """Run backtest using the simple strategy"""
        # Generate signals
        self.generate_signals()
        
        # Skip the initial period where indicators are being calculated
        start_index = max(self.settings['short_window'], self.settings['long_window'])
        
        for i in range(start_index, len(self.df)):
            date = self.df.index[i]
            row = self.df.iloc[i]
            price = row['close']
            signal = row['positions']
            
            if signal == 1:  # BUY signal
                if self.balance >= self.settings['fixed_position_size']:
                    # Use fixed position size instead of percentage
                    amount_to_invest = min(self.settings['fixed_position_size'], self.balance)
                    coins_bought = amount_to_invest / price
                    self.position += coins_bought
                    self.trades.append((date, 'BUY', price, amount_to_invest))
                    self.balance -= amount_to_invest
            
            elif signal == -1:  # SELL signal
                if self.position > 0:
                    # Sell all coins
                    sell_value = self.position * price
                    self.trades.append((date, 'SELL', price, sell_value))
                    self.balance += sell_value
                    self.position = 0
        
        # Final balance (if holding position at end)
        if self.position > 0:
            final_price = self.df.iloc[-1]['close']
            final_value = self.position * final_price
            self.trades.append((self.df.index[-1], 'FINAL_SELL', final_price, final_value))
            self.balance += final_value
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
                'win_rate': 0
            }
        
        initial_balance = self.settings['initial_balance']
        final_balance = self.balance
        profit = final_balance - initial_balance
        profit_percent = (profit / initial_balance) * 100
        
        # Calculate trade statistics
        buy_trades = [t for t in self.trades if t[1] == 'BUY']
        sell_trades = [t for t in self.trades if t[1] in ('SELL', 'FINAL_SELL')]
        
        # Pair buy and sell trades to determine winners and losers
        trade_pairs = []
        current_position = 0
        current_cost = 0
        
        for trade in self.trades:
            date, action, price, amount = trade
            
            if action == 'BUY':
                # Add to position using average cost
                new_coins = amount / price
                if current_position == 0:
                    current_cost = price
                else:
                    # Calculate weighted average cost
                    current_cost = ((current_position * current_cost) + amount) / (current_position + new_coins)
                current_position += new_coins
            
            elif action in ('SELL', 'FINAL_SELL'):
                # Calculate profit for this trade
                if current_position > 0:
                    profit_per_coin = price - current_cost
                    trade_pairs.append({
                        'buy_price': current_cost,
                        'sell_price': price,
                        'profit': (price - current_cost) / current_cost * 100
                    })
                    current_position = 0
                    current_cost = 0
        
        # Calculate win rate
        winning_trades = [t for t in trade_pairs if t['profit'] > 0]
        success_trades = len(winning_trades)
        failed_trades = len(trade_pairs) - success_trades
        win_rate = success_trades / len(trade_pairs) if trade_pairs else 0
        
        return {
            'final_balance': final_balance,
            'profit': profit,
            'profit_percent': profit_percent,
            'success_trades': success_trades,
            'failed_trades': failed_trades,
            'total_trades': len(trade_pairs),
            'win_rate': win_rate * 100  # as percentage
        }
