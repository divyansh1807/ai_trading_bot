import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import sys
from datetime import datetime

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import with error handling
try:
    from bot.simple_strategy import SimpleStrategy
    BOT_AVAILABLE = True
except ImportError:
    BOT_AVAILABLE = False

try:
    from utils.data_loader import save_yearly_data, get_latest_data
    DATA_LOADER_AVAILABLE = True
except ImportError:
    DATA_LOADER_AVAILABLE = False

def run_gui():
    # Create the main window
    root = tk.Tk()
    root.title("AI Crypto Trading Bot")

    # Set the window size and configure
    root.geometry("1000x700")
    root.configure(bg="#2c3e50")

    # Create a style for the widgets
    style = ttk.Style()
    style.theme_use('clam')  # Use a modern theme

    # Configure styles
    style.configure("TFrame", background="#f0f0f0")
    style.configure("TLabelframe", background="#f0f0f0")
    style.configure("TLabelframe.Label", font=("Arial", 12, "bold"), background="#f0f0f0")
    style.configure("TButton", font=("Arial", 11), background="#3498db")
    style.configure("TLabel", font=("Arial", 11), background="#f0f0f0")
    style.configure("TCombobox", font=("Arial", 11))
    style.configure("Header.TLabel", font=("Arial", 20, "bold"), foreground="#ffffff", background="#2c3e50")
    style.configure("Subheader.TLabel", font=("Arial", 14), foreground="#ffffff", background="#2c3e50")

    # Create a header frame
    header_frame = ttk.Frame(root)
    header_frame.pack(fill="x", pady=0)
    header_frame.configure(style="")

    # Add a custom header with background color
    header_bg = tk.Frame(header_frame, bg="#2c3e50", height=80)
    header_bg.pack(fill="x")

    # Create a header
    header_label = ttk.Label(header_bg, text="AI Crypto Trading Bot", style="Header.TLabel")
    header_label.pack(pady=10, padx=20, anchor="w")

    # Add a subtitle
    subtitle_label = ttk.Label(header_bg, text="Backtest your trading strategies on historical data", style="Subheader.TLabel")
    subtitle_label.pack(padx=20, anchor="w")

    # Create a main frame
    main_frame = ttk.Frame(root, padding=20)
    main_frame.pack(fill="both", expand=True)

    # Create control panel frame with scrolling
    control_outer_frame = ttk.LabelFrame(main_frame, text="Control Panel", padding=10)
    control_outer_frame.pack(side="left", fill="y", padx=10)

    # Add a canvas and scrollbar for scrolling
    control_canvas = tk.Canvas(control_outer_frame, borderwidth=0, width=200)
    control_scrollbar = ttk.Scrollbar(control_outer_frame, orient="vertical", command=control_canvas.yview)
    control_frame = ttk.Frame(control_canvas)

    # Configure the canvas
    control_canvas.configure(yscrollcommand=control_scrollbar.set)
    control_canvas.pack(side="left", fill="both", expand=True)
    control_scrollbar.pack(side="right", fill="y")

    # Add the frame to the canvas
    control_canvas_frame_id = control_canvas.create_window((0, 0), window=control_frame, anchor="nw")

    # Configure scrolling
    def configure_control_scroll_region(event):
        control_canvas.configure(scrollregion=control_canvas.bbox("all"))

    def configure_control_canvas_width(event):
        control_canvas.itemconfig(control_canvas_frame_id, width=event.width)

    control_frame.bind("<Configure>", configure_control_scroll_region)
    control_canvas.bind("<Configure>", configure_control_canvas_width)

    # Create results frame with scrolling
    results_outer_frame = ttk.LabelFrame(main_frame, text="Results", padding=10)
    results_outer_frame.pack(side="right", fill="both", expand=True, padx=10)

    # Add a canvas and scrollbar for scrolling
    results_canvas = tk.Canvas(results_outer_frame, borderwidth=0)
    results_scrollbar = ttk.Scrollbar(results_outer_frame, orient="vertical", command=results_canvas.yview)
    results_frame = ttk.Frame(results_canvas)

    # Configure the canvas
    results_canvas.configure(yscrollcommand=results_scrollbar.set)
    results_canvas.pack(side="left", fill="both", expand=True)
    results_scrollbar.pack(side="right", fill="y")

    # Add the frame to the canvas
    canvas_frame_id = results_canvas.create_window((0, 0), window=results_frame, anchor="nw")

    # Configure scrolling
    def configure_scroll_region(event):
        results_canvas.configure(scrollregion=results_canvas.bbox("all"))

    def configure_canvas_width(event):
        results_canvas.itemconfig(canvas_frame_id, width=event.width)

    results_frame.bind("<Configure>", configure_scroll_region)
    results_canvas.bind("<Configure>", configure_canvas_width)

    # Dropdown for selecting cryptocurrency (BTC or ETH)
    crypto_label = ttk.Label(control_frame, text="Select Cryptocurrency")
    crypto_label.pack(pady=10, anchor="w")
    crypto_var = tk.StringVar()
    crypto_dropdown = ttk.Combobox(control_frame, textvariable=crypto_var, values=["BTC", "ETH"], width=15)
    crypto_dropdown.pack(pady=5)
    crypto_dropdown.current(0)  # Set default to BTC

    # Dropdown for selecting the year
    year_label = ttk.Label(control_frame, text="Select Year")
    year_label.pack(pady=10, anchor="w")
    year_var = tk.StringVar()
    year_dropdown = ttk.Combobox(control_frame, textvariable=year_var,
                               values=["2020", "2021", "2022", "2023", "2024", "2025"], width=15)
    year_dropdown.pack(pady=5)
    year_dropdown.current(3)  # Set default to 2023

    # Strategy options
    strategy_label = ttk.Label(control_frame, text="Strategy Options")
    strategy_label.pack(pady=10, anchor="w")

    # Checkboxes for strategy options
    use_ml_var = tk.BooleanVar(value=True)
    use_ml_check = ttk.Checkbutton(control_frame, text="Use Machine Learning", variable=use_ml_var)
    use_ml_check.pack(pady=5, anchor="w")

    use_sentiment_var = tk.BooleanVar(value=True)
    use_sentiment_check = ttk.Checkbutton(control_frame, text="Use Sentiment Analysis", variable=use_sentiment_var)
    use_sentiment_check.pack(pady=5, anchor="w")

    # Create a notebook (tabbed interface) for results
    results_notebook = ttk.Notebook(results_frame)
    results_notebook.pack(fill="both", expand=True, pady=10)

    # Tab 1: Performance Graph
    graph_tab = ttk.Frame(results_notebook)
    results_notebook.add(graph_tab, text="Performance Graph")

    # Canvas for displaying the performance graph
    canvas_frame = ttk.Frame(graph_tab)
    canvas_frame.pack(pady=10, fill="both", expand=True)
    canvas = None

    # Tab 2: Performance Summary
    summary_tab = ttk.Frame(results_notebook)
    results_notebook.add(summary_tab, text="Performance Summary")

    # Add scrollable text widget for summary
    summary_text = tk.Text(summary_tab, wrap=tk.WORD, height=8, width=40, font=("Arial", 10))
    summary_scrollbar = ttk.Scrollbar(summary_tab, orient="vertical", command=summary_text.yview)
    summary_text.configure(yscrollcommand=summary_scrollbar.set)

    summary_text.pack(side="left", fill="both", expand=True, pady=5)
    summary_scrollbar.pack(side="right", fill="y")

    # Insert initial text
    summary_text.insert(tk.END, "Run a backtest to see results")
    summary_text.config(state=tk.DISABLED)  # Make read-only

    # Tab 3: Trade Details
    trades_tab = ttk.Frame(results_notebook)
    results_notebook.add(trades_tab, text="Trade Details")

    # Create a frame for the trades table
    trades_frame = ttk.Frame(trades_tab)
    trades_frame.pack(fill="both", expand=True, padx=5, pady=5)

    # Create Treeview for trades table
    trades_columns = ("date", "action", "price", "amount", "balance", "profit")
    trades_table = ttk.Treeview(trades_frame, columns=trades_columns, show="headings")

    # Configure tag colors for different trade types
    trades_table.tag_configure('buy', background='#ffdddd')  # Light red for buys
    trades_table.tag_configure('sell', background='#ddffdd')  # Light green for sells
    trades_table.tag_configure('final_sell', background='#ddffee')  # Light blue-green for final sells
    trades_table.tag_configure('stop_loss', background='#ffcccc')  # Darker red for stop losses

    # Define column headings
    trades_table.heading("date", text="Date")
    trades_table.heading("action", text="Action")
    trades_table.heading("price", text="Price ($)")
    trades_table.heading("amount", text="Amount ($)")
    trades_table.heading("balance", text="Balance ($)")
    trades_table.heading("profit", text="Profit/Loss ($)")

    # Define column widths
    trades_table.column("date", width=150)
    trades_table.column("action", width=80)
    trades_table.column("price", width=100)
    trades_table.column("amount", width=100)
    trades_table.column("balance", width=100)
    trades_table.column("profit", width=100)

    # Add scrollbars to the treeview
    trades_y_scroll = ttk.Scrollbar(trades_frame, orient="vertical", command=trades_table.yview)
    trades_x_scroll = ttk.Scrollbar(trades_frame, orient="horizontal", command=trades_table.xview)
    trades_table.configure(yscrollcommand=trades_y_scroll.set, xscrollcommand=trades_x_scroll.set)

    # Pack the scrollbars and treeview
    trades_y_scroll.pack(side="right", fill="y")
    trades_x_scroll.pack(side="bottom", fill="x")
    trades_table.pack(side="left", fill="both", expand=True)

    # Insert initial message
    trades_table.insert("", "end", values=("Run a backtest to see trade details", "", "", "", "", ""))

    # Add a summary frame below the trades table
    trades_summary_frame = ttk.LabelFrame(trades_tab, text="Trade Summary", padding=10)
    trades_summary_frame.pack(fill="x", padx=5, pady=5)

    # Create a grid for summary statistics
    trades_summary_grid = ttk.Frame(trades_summary_frame)
    trades_summary_grid.pack(fill="x", padx=5, pady=5)

    # Create labels for summary statistics
    ttk.Label(trades_summary_grid, text="Total Trades:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
    ttk.Label(trades_summary_grid, text="Buy Trades:").grid(row=0, column=2, sticky="w", padx=5, pady=2)
    ttk.Label(trades_summary_grid, text="Sell Trades:").grid(row=1, column=0, sticky="w", padx=5, pady=2)
    ttk.Label(trades_summary_grid, text="Profitable Trades:").grid(row=1, column=2, sticky="w", padx=5, pady=2)
    ttk.Label(trades_summary_grid, text="Total Profit/Loss:").grid(row=2, column=0, sticky="w", padx=5, pady=2)
    ttk.Label(trades_summary_grid, text="Win Rate:").grid(row=2, column=2, sticky="w", padx=5, pady=2)

    # Create variables for summary statistics
    total_trades_var = tk.StringVar(value="-")
    buy_trades_var = tk.StringVar(value="-")
    sell_trades_var = tk.StringVar(value="-")
    profitable_trades_var = tk.StringVar(value="-")
    total_profit_var = tk.StringVar(value="-")
    win_rate_var = tk.StringVar(value="-")

    # Create labels to display the values
    ttk.Label(trades_summary_grid, textvariable=total_trades_var, font=("Arial", 10, "bold")).grid(row=0, column=1, sticky="w", padx=5, pady=2)
    ttk.Label(trades_summary_grid, textvariable=buy_trades_var, font=("Arial", 10, "bold")).grid(row=0, column=3, sticky="w", padx=5, pady=2)
    ttk.Label(trades_summary_grid, textvariable=sell_trades_var, font=("Arial", 10, "bold")).grid(row=1, column=1, sticky="w", padx=5, pady=2)
    ttk.Label(trades_summary_grid, textvariable=profitable_trades_var, font=("Arial", 10, "bold")).grid(row=1, column=3, sticky="w", padx=5, pady=2)
    ttk.Label(trades_summary_grid, textvariable=total_profit_var, font=("Arial", 10, "bold")).grid(row=2, column=1, sticky="w", padx=5, pady=2)
    ttk.Label(trades_summary_grid, textvariable=win_rate_var, font=("Arial", 10, "bold")).grid(row=2, column=3, sticky="w", padx=5, pady=2)

    # Status bar
    status_var = tk.StringVar()
    status_var.set("Ready")
    status_bar = ttk.Label(root, textvariable=status_var, relief=tk.SUNKEN, anchor=tk.W)
    status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    # Function to plot the balance performance graph
    def plot_performance(df):
        nonlocal canvas
        if canvas:
            canvas.get_tk_widget().destroy()

        fig, ax = plt.subplots(figsize=(8, 5), dpi=100)
        ax.plot(df['Date'], df['Balance'], label="Balance over Time", color='#1f77b4', linewidth=2)
        ax.set_xlabel('Date')
        ax.set_ylabel('Balance ($)')
        ax.set_title('Balance Performance Over Time')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()

        # Render the plot on Tkinter window
        canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    # Function to run the backtest
    def run_backtest():
        symbol = crypto_var.get().lower()
        year = year_var.get()

        if not symbol or not year:
            messagebox.showerror("Error", "Please select both cryptocurrency and year")
            return

        status_var.set(f"Running backtest for {symbol.upper()} {year}...")
        root.update()

        file_path = f"data/{symbol}_{year}_trades.csv"
        if not os.path.exists(file_path):
            messagebox.showerror("Error", f"Backtest data for {symbol.upper()} {year} not found.")
            status_var.set("Ready")
            return

        try:
            # Get initial balance from entry
            try:
                initial_balance = float(balance_var.get())
                if initial_balance <= 0:
                    initial_balance = 10000  # Default if negative or zero
                    balance_var.set("10000")
            except ValueError:
                initial_balance = 10000  # Default if invalid input
                balance_var.set("10000")

            # Create settings dictionary with the initial balance
            settings = {
                'initial_balance': initial_balance,
                'fixed_position_size': initial_balance * 0.1  # Use 10% of initial balance as fixed position size
            }

            # Load historical price data
            data_file = f"data/{symbol}_{year}.csv"
            if not os.path.exists(data_file):
                messagebox.showerror("Error", f"Price data for {symbol.upper()} {year} not found.")
                status_var.set("Ready")
                return

            # Load the data and run the backtest
            price_data = pd.read_csv(data_file, index_col='timestamp', parse_dates=True)

            # Create the strategy and run backtest
            strategy = SimpleStrategy(price_data, settings)
            trades, final_balance = strategy.run_backtest()
            performance = strategy.get_performance()

            # Create a DataFrame for visualization
            df = pd.DataFrame(trades, columns=['Date', 'Action', 'Price', 'Amount'])
            df['Balance'] = 0

            # Calculate running balance
            running_balance = initial_balance
            for i, row in df.iterrows():
                if row['Action'] == 'BUY':
                    running_balance -= row['Amount']
                else:  # SELL or FINAL_SELL
                    running_balance += row['Amount']
                df.at[i, 'Balance'] = running_balance

            # Save the results for future reference
            df.to_csv(file_path, index=False)

            # Extract performance metrics
            total_trades = performance['total_trades']
            successful_trades = performance['success_trades']
            loss_trades = performance['failed_trades']
            final_balance = performance['final_balance']
            profit = performance['profit']
            profit_percent = performance['profit_percent']
            win_rate = performance['win_rate']

            # Display a summary of results
            result_text = f"Symbol: {symbol.upper()} | Year: {year}\n"
            result_text += f"Total Trades: {total_trades}\n"
            result_text += f"Successful Trades: {successful_trades}\n"
            result_text += f"Loss Trades: {loss_trades}\n"
            result_text += f"Final Balance: ${final_balance:.2f}\n"
            result_text += f"Profit/Loss: ${profit:.2f} ({profit_percent:.2f}%)\n"

            result_text += f"Win Rate: {win_rate:.2f}%\n"

            # Update the summary text widget
            summary_text.config(state=tk.NORMAL)  # Make writable
            summary_text.delete(1.0, tk.END)     # Clear existing text
            summary_text.insert(tk.END, result_text)
            summary_text.config(state=tk.DISABLED)  # Make read-only again

            # Plot the balance change over time
            plot_performance(df)

            # Update the trades table
            # First clear existing data
            for item in trades_table.get_children():
                trades_table.delete(item)

            # Add new trade data
            last_buy_price = 0
            last_buy_amount = 0

            for i, row in df.iterrows():
                date_str = row['Date'].strftime('%Y-%m-%d %H:%M:%S') if isinstance(row['Date'], pd.Timestamp) else str(row['Date'])
                action = row['Action']
                price = f"${row['Price']:.2f}"

                # Format amount based on action (negative for BUY, positive for SELL)
                if action == 'BUY':
                    amount_str = f"-${row['Amount']:.2f}"
                    profit_str = ""
                    last_buy_price = row['Price']
                    last_buy_amount = row['Amount']
                else:  # SELL or FINAL_SELL
                    amount_str = f"+${row['Amount']:.2f}"
                    # Calculate profit/loss for this trade
                    if last_buy_price > 0:
                        profit = row['Amount'] - last_buy_amount
                        profit_str = f"${profit:.2f}"
                        # Color code profit/loss
                        if profit > 0:
                            profit_str = f"+{profit_str}"
                        elif profit < 0:
                            profit_str = f"{profit_str}"
                    else:
                        profit_str = "N/A"

                balance = f"${row['Balance']:.2f}"

                # Determine the tag based on the action
                if action == 'BUY':
                    tag = 'buy'
                elif action == 'SELL':
                    tag = 'sell'
                elif action == 'FINAL_SELL':
                    tag = 'final_sell'
                elif action == 'STOP_LOSS':
                    tag = 'stop_loss'
                else:
                    tag = ''

                # Insert with the appropriate tag
                trades_table.insert("", "end", values=(date_str, action, price, amount_str, balance, profit_str), tags=(tag,))

            # Update trade summary statistics
            total_trades = len(df)
            buy_trades = len(df[df['Action'] == 'BUY'])
            sell_trades = total_trades - buy_trades
            profitable_trades = performance['success_trades']
            total_profit = performance['profit']
            win_rate = performance['win_rate']

            # Update the summary variables
            total_trades_var.set(str(total_trades))
            buy_trades_var.set(str(buy_trades))
            sell_trades_var.set(str(sell_trades))
            profitable_trades_var.set(str(profitable_trades))
            total_profit_var.set(f"${total_profit:.2f}")
            win_rate_var.set(f"{win_rate:.2f}%")

            # Switch to the trades tab to show the details
            results_notebook.select(2)  # Index 2 is the trades tab

            status_var.set(f"Backtest completed for {symbol.upper()} {year}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process backtest data: {str(e)}")
            status_var.set("Ready")

    # Add advanced options section
    advanced_frame = ttk.LabelFrame(control_frame, text="Advanced Options")
    advanced_frame.pack(pady=10, fill="x")

    # Add initial balance input
    balance_frame = ttk.Frame(advanced_frame)
    balance_frame.pack(pady=5, fill="x")

    balance_label = ttk.Label(balance_frame, text="Initial Balance:")
    balance_label.pack(side="left", padx=5)

    balance_var = tk.StringVar(value="10000")
    balance_entry = ttk.Entry(balance_frame, textvariable=balance_var, width=10)
    balance_entry.pack(side="right", padx=5)

    # Add buttons frame
    buttons_frame = ttk.Frame(control_frame)
    buttons_frame.pack(pady=20, fill="x")

    # Button to run the backtest
    run_button = ttk.Button(buttons_frame, text="Run Backtest", command=run_backtest)
    run_button.pack(side="left", padx=5, expand=True, fill="x")

    # Button to clear results
    def clear_results():
        # Clear summary text
        summary_text.config(state=tk.NORMAL)
        summary_text.delete(1.0, tk.END)
        summary_text.insert(tk.END, "Run a backtest to see results")
        summary_text.config(state=tk.DISABLED)

        # Clear chart
        if canvas:
            canvas.get_tk_widget().destroy()

        # Clear trades table
        for item in trades_table.get_children():
            trades_table.delete(item)
        trades_table.insert("", "end", values=("Run a backtest to see trade details", "", "", "", "", ""))

        # Reset trade summary statistics
        total_trades_var.set("-")
        buy_trades_var.set("-")
        sell_trades_var.set("-")
        profitable_trades_var.set("-")
        total_profit_var.set("-")
        win_rate_var.set("-")

        # Reset status
        status_var.set("Ready")

    clear_button = ttk.Button(buttons_frame, text="Clear Results", command=clear_results)
    clear_button.pack(side="right", padx=5, expand=True, fill="x")

    # Run the GUI
    root.mainloop()

    return root
