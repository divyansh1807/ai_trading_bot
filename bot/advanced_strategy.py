import pandas as pd
import numpy as np

class AdvancedTradingBot:
    def __init__(self, df, settings=None):
        self.df = df.copy()
        
        # Default settings
        self.settings = {
            'sma_short': 20,
            'sma_long': 50,
            'rsi_period': 14,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'initial_balance': 10000,
            'position_size': 1.0,  # Percentage of balance to use (1.0 = 100%)
            'stop_loss': 0.05      # 5% stop loss
        }
        
        # Update with custom settings if provided
        if settings:
            self.settings.update(settings)
            
        self.balance = self.settings['initial_balance']
        self.position = 0  # Number of coins
        self.trades = []
        self.current_stop_loss = 0
    
    def calculate_indicators(self):
        # SMA indicators
        self.df['SMA_short'] = self.df['close'].rolling(window=self.settings['sma_short']).mean()
        self.df['SMA_long'] = self.df['close'].rolling(window=self.settings['sma_long']).mean()
        
        # RSI calculation
        delta = self.df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=self.settings['rsi_period']).mean()
        avg_loss = loss.rolling(window=self.settings['rsi_period']).mean()
        
        rs = avg_gain / avg_loss
        self.df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        self.df['EMA12'] = self.df['close'].ewm(span=12, adjust=False).mean()
        self.df['EMA26'] = self.df['close'].ewm(span=26, adjust=False).mean()
        self.df['MACD'] = self.df['EMA12'] - self.df['EMA26']
        self.df['Signal_Line'] = self.df['MACD'].ewm(span=9, adjust=False).mean()
        self.df['MACD_Histogram'] = self.df['MACD'] - self.df['Signal_Line']
        
        # Bollinger Bands
        self.df['20MA'] = self.df['close'].rolling(window=20).mean()
        self.df['20STD'] = self.df['close'].rolling(window=20).std()
        self.df['Upper_Band'] = self.df['20MA'] + (self.df['20STD'] * 2)
        self.df['Lower_Band'] = self.df['20MA'] - (self.df['20STD'] * 2)
        
        # Generate signals
        self.generate_signals()
    
    def generate_signals(self):
        # Initialize signal columns
        self.df['sma_signal'] = 0
        self.df['rsi_signal'] = 0
        self.df['macd_signal'] = 0
        self.df['bb_signal'] = 0
        self.df['combined_signal'] = 0
        
        # SMA crossover signal
        self.df.loc[self.df.index[self.settings['sma_short']:], 'sma_signal'] = np.where(
            self.df['SMA_short'][self.settings['sma_short']:] > self.df['SMA_long'][self.settings['sma_short']:], 1, -1)
        
        # RSI signal
        self.df.loc[self.df['RSI'] < self.settings['rsi_oversold'], 'rsi_signal'] = 1
        self.df.loc[self.df['RSI'] > self.settings['rsi_overbought'], 'rsi_signal'] = -1
        
        # MACD signal
        self.df.loc[self.df['MACD'] > self.df['Signal_Line'], 'macd_signal'] = 1
        self.df.loc[self.df['MACD'] < self.df['Signal_Line'], 'macd_signal'] = -1
        
        # Bollinger Bands signal
        self.df.loc[self.df['close'] < self.df['Lower_Band'], 'bb_signal'] = 1
        self.df.loc[self.df['close'] > self.df['Upper_Band'], 'bb_signal'] = -1
        
        # Combined signal (simple majority voting)
        signals = self.df[['sma_signal', 'rsi_signal', 'macd_signal', 'bb_signal']]
        self.df['combined_signal'] = signals.sum(axis=1)
        
        # Generate positions (1 for buy, -1 for sell, 0 for hold)
        self.df['positions'] = 0
        self.df.loc[self.df['combined_signal'] >= 2, 'positions'] = 1
        self.df.loc[self.df['combined_signal'] <= -2, 'positions'] = -1
        
        # Convert to actual trade signals (entry/exit points)
        self.df['trade_signal'] = self.df['positions'].diff()
    
    def run_backtest(self):
        self.calculate_indicators()
        
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
            signal = row['trade_signal']
            
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
                'sharpe_ratio': 0
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
            'sharpe_ratio': sharpe_ratio
        }
