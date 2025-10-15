# LOBSTER Market Data Analyzer

LOBSTER (Limit Order Book System - The Efficient Reconstructor) market data parser and visualizer.

Work in progress: Backtesting engine

## Setup

```bash
git clone <repo-url>
cd lobster
uv sync

streamlit run gui_app.py # local data

streamlit run gui_app.py  # huggingface data

```

## Installation

```bash
uv sync
```

**Dependencies:** polars, streamlit, plotly, huggingface-hub

## Usage

### GUI Visualization

```bash
streamlit run gui_app.py
```

Navigate between three views:
- **L1**: Price candlesticks, volume profiles, execution markers
- **L2**: Aggregated depth charts by price level
- **L3**: Individual order tracking with FIFO queue positions

## HuggingFace Setup

**Configuration**

Create `.streamlit/secrets.toml`:
```toml
HF_REPO_ID = "USERNAME/lobster-data"
```

**Sync All Datasets**

The `sync_datasets.py` script handles everything: creating the repository, downloading missing datasets from lobsterdata.com, uploading to HuggingFace, and generating the README.

```bash
huggingface-cli login

python sync_datasets.py --repo-id USERNAME/lobster-data

python sync_datasets.py --repo-id USERNAME/lobster-data --dry-run  # Preview only
python sync_datasets.py --repo-id USERNAME/lobster-data --tickers AAPL MSFT  # Specific tickers
python sync_datasets.py --repo-id USERNAME/lobster-data --private  # Private repo
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

## Sample Data

Sample datasets are available at [lobsterdata.com/info/DataSamples.php](https://lobsterdata.com/info/DataSamples.php):
