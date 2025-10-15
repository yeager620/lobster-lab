# LOBSTER Order Book Visualizer

LOBSTER order book data visualizer

GUI for visualizing limit order book data with L1, L2, and L3 views

## Setup

```bash
uv sync
uv run streamlit run gui_app.py
```

Three pages:
- **L1**: Price time series, candlesticks, volume profiles
- **L2**: Order book with liquidity aggregated by price
- **L3**: Market by order with FIFO queue positions

## Configuration

### HuggingFace

By default, the GUI loads from `totalorganfailure/lobster-data`. To use a different repository:

```bash
export HF_REPO_ID="your-username/your-repo"
uv run streamlit run gui_app.py
```

Or create `.streamlit/secrets.toml`:
```toml
HF_REPO_ID = "your-username/your-repo"
```

### Local Files

```bash
export HF_REPO_ID=""
uv run streamlit run gui_app.py
```

Local files must be in `./LOBSTER_SampleFile_*` directories

## Sync Datasets to HuggingFace


```bash
uv run huggingface-cli login

uv run python sync_datasets.py --repo-id your-username/lobster-data

uv run python sync_datasets.py --repo-id your-username/lobster-data --dry-run
uv run python sync_datasets.py --repo-id your-username/lobster-data --tickers AAPL MSFT
uv run python sync_datasets.py --repo-id your-username/lobster-data --force-readme --keep-local
uv run python sync_datasets.py --repo-id your-username/lobster-data --private
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