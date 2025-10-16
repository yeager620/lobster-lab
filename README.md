# LOBSTER Order Book Visualizer

Interactive visualization and backtesting toolkit for LOBSTER limit order book data with L1, L2, and L3 market microstructure views.

[**Launch Streamlit App**](https://lobster-lab.streamlit.app/)

[**Hugging Face Spaces**](https://huggingface.co/spaces/totalorganfailure/lobster-lab)

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
# use default HuggingFace dataset
uv run streamlit run gui_app.py

# use custom HuggingFace dataset
export HF_REPO_ID="USERNAME/REPO_NAME"
uv run streamlit run gui_app.py

# use local files (must be in ./LOBSTER_SampleFile_*)
export HF_REPO_ID=""
uv run streamlit run gui_app.py
```

### Streamlit Secrets

Create `.streamlit/secrets.toml`:
```toml
HF_REPO_ID = "USERNAME/REPO_NAME"
```

## Data Management

### Sync Datasets to HuggingFace

```bash
uv run huggingface-cli login
uv run python scripts/sync_datasets.py --repo-id your-username/lobster-lab-data
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
