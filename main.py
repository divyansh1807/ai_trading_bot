import os
import sys
import tkinter as tk
from tkinter import messagebox

def launch_gui():
    print("Launching AI Crypto Trading Bot GUI...")
    try:
        # Import the GUI module
        from gui.gui import run_gui

        # Run the GUI
        print("Starting GUI...")
        run_gui()
        print("GUI closed.")
    except ImportError as e:
        messagebox.showerror("Error", f"Failed to load GUI: {str(e)}")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")

if __name__ == "__main__":
    # Launch the GUI
    launch_gui()