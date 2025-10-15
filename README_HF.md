---
title: LOBSTER Order Book Visualizer
emoji: 🦞
colorFrom: red
colorTo: blue
sdk: streamlit
sdk_version: "1.50.0"
python_version: "3.13"
app_file: gui_app.py
pinned: false
license: bsd-3-clause
---

# LOBSTER Order Book Visualizer

Interactive visualization tool for LOBSTER high-frequency market data.

## Features

- **L1 Visualization**: Price action, candlestick charts, and volume profiles
- **L2 Visualization**: Price-level order book depth with aggregated liquidity
- **L3 Visualization**: Order-level microstructure and event flow
- **Tickers**: AAPL, MSFT, GOOG, INTC, AMZN, SPY (1-50 levels)

## Data

Datasets are loaded from [totalorganfailure/lobster-data](https://huggingface.co/datasets/totalorganfailure/lobster-data).

All data is from June 21, 2012, NASDAQ TotalView-ITCH.

## GitHub Source

Source code: [github.com/yeager620/lobster-lab](https://github.com/yeager620/lobster-lab)
