# LOBSTER Order Book Visualizer

Interactive visualization and backtesting toolkit for LOBSTER limit order book data with L1, L2, and L3 market microstructure views.

## Features

- **L1 View**: Price action with candlesticks, volume profiles, and execution analysis
- **L2 View**: Aggregated order book depth by price level
- **L3 View**: Market-by-order with FIFO queue visualization
- **Backtesting**: Simple market-making strategy backtester
- **Data Sync**: HuggingFace dataset synchronization utilities

## Quick Start

### Installation

```bash
uv sync
```

### Run GUI

```bash
# Run with default HuggingFace dataset (totalorganfailure/lobster-lab-data)
uv run streamlit run gui_app.py

# Run with custom HuggingFace dataset
export HF_REPO_ID="your-username/your-repo"
uv run streamlit run gui_app.py

# Run with local files (must be in ./LOBSTER_SampleFile_* directories)
export HF_REPO_ID=""
uv run streamlit run gui_app.py
```

### Run Backtesting

```bash
# Default: AAPL with 100 shares quote size
uv run python main.py

# Multiple tickers with custom parameters
uv run python main.py --tickers AAPL MSFT SPY --quote-size 200 --participation 0.2
```

## Project Structure

```
lobster/
├── gui_app.py              # Streamlit GUI entry point
├── gui_pages/              # GUI page modules
│   ├── l1_page.py         # Price action view
│   ├── l2_page.py         # Order book depth view
│   ├── l3_page.py         # Market-by-order view
│   └── shared.py          # Shared utilities
├── lobster_parsing.py      # Core LOBSTER data parser
├── backtesting.py          # Market-making backtest engine
├── main.py                 # Backtesting CLI entry point
└── scripts/
    └── sync_datasets.py    # HuggingFace sync utility
```

## Configuration

### Streamlit Secrets

Create `.streamlit/secrets.toml`:
```toml
HF_REPO_ID = "your-username/your-repo"
```

### Environment Variables

```bash
export HF_REPO_ID="your-username/lobster-data"  # HuggingFace dataset repo
```

## Data Management

### Sync Datasets to HuggingFace

```bash
# Login to HuggingFace
uv run huggingface-cli login

# Sync all datasets
uv run python scripts/sync_datasets.py --repo-id your-username/lobster-lab-data

# Dry run to preview changes
uv run python scripts/sync_datasets.py --repo-id your-username/lobster-lab-data --dry-run

# Sync specific tickers
uv run python scripts/sync_datasets.py --repo-id your-username/lobster-lab-data --tickers AAPL MSFT

# Update README only
uv run python scripts/sync_datasets.py --repo-id your-username/lobster-lab-data --force-readme

# Keep local files after upload
uv run python scripts/sync_datasets.py --repo-id your-username/lobster-lab-data --keep-local

# Create private dataset
uv run python scripts/sync_datasets.py --repo-id your-username/lobster-lab-data --private
```

## Data Format

### Message File (Nx6)
| Column | Description |
|--------|-------------|
| time | Seconds after midnight (decimal) |
| type | 1=New, 2=Cancel, 3=Delete, 4=Exec(Visible), 5=Exec(Hidden), 7=Halt |
| order\_id | Unique order identifier |
| size | Shares |
| price | Dollars × 10,000 (e.g., $91.14 → 911400) |
| direction | -1=Sell, 1=Buy |

### Orderbook File (Nx4L)
For each level i: `[ask_price_i, ask_size_i, bid_price_i, bid_size_i]`

Unoccupied levels: `ask_price=9999999999, bid_price=-9999999999, size=0`