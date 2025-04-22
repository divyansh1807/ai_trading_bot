import pandas as pd

class TradingBot:
    def __init__(self, df, short_window=20, long_window=50, initial_balance=10000):
        self.df = df.copy()
        self.short_window = short_window
        self.long_window = long_window
        self.balance = initial_balance
        self.position = 0  # Number of coins
        self.trades = []

    def generate_signals(self):
        self.df['SMA_short'] = self.df['close'].rolling(window=self.short_window).mean()
        self.df['SMA_long'] = self.df['close'].rolling(window=self.long_window).mean()

        self.df['signal'] = 0
        # Use .loc[] to avoid chained assignment warning
        self.df.loc[self.df.index[self.short_window:], 'signal'] = \
            (self.df['SMA_short'][self.short_window:] > self.df['SMA_long'][self.short_window:]).astype(int)
        self.df['positions'] = self.df['signal'].diff()

    def run_backtest(self):
        self.generate_signals()

        for date, row in self.df.iterrows():
            price = row['close']
            signal = row['positions']

            if signal == 1:  # BUY
                if self.balance > 0:
                    self.position = self.balance / price
                    self.trades.append((date, 'BUY', price, self.balance))
                    self.balance = 0

            elif signal == -1:  # SELL
                if self.position > 0:
                    self.balance = self.position * price
                    self.trades.append((date, 'SELL', price, self.balance))
                    self.position = 0

        # Final balance (if holding position at end)
        if self.position > 0:
            final_price = self.df.iloc[-1]['close']
            self.balance = self.position * final_price
            self.trades.append((self.df.index[-1], 'FINAL SELL', final_price, self.balance))
            self.position = 0

        return self.trades, self.balance

    def get_performance(self):
        profit = self.balance - 10000
        success_trades = [t for t in self.trades if t[1] == 'SELL' and t[3] > 10000]
        failed_trades = [t for t in self.trades if t[1] == 'SELL' and t[3] <= 10000]

        return {
            'final_balance': self.balance,
            'profit': profit,
            'success_trades': len(success_trades),
            'failed_trades': len(failed_trades),
            'total_trades': len(self.trades)
        }
