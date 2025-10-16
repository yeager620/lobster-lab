import streamlit as st
from pathlib import Path
from lobster_parsing import read_lobster
from typing import Tuple, Any, Iterable
import os
from datetime import datetime, timedelta
import zoneinfo

HF_REPO_ID = os.getenv("HF_REPO_ID", "totalorganfailure/lobster-data")

if not HF_REPO_ID and hasattr(st, "secrets") and "HF_REPO_ID" in st.secrets:
    HF_REPO_ID = st.secrets["HF_REPO_ID"]

USE_HUGGINGFACE = bool(HF_REPO_ID)

if USE_HUGGINGFACE:
    from huggingface_hub import hf_hub_download


_GLOBAL_STYLE = """
<style>
/* Establish responsive typography that scales with viewport width */
:root {
    --lobster-lab-h1: clamp(2.0rem, 2.4vw + 0.8rem, 3.2rem);
    --lobster-lab-h2: clamp(1.6rem, 1.9vw + 0.6rem, 2.4rem);
    --lobster-lab-h3: clamp(1.35rem, 1.4vw + 0.55rem, 1.9rem);
    --lobster-lab-text: clamp(0.95rem, 0.95vw + 0.55rem, 1.1rem);
}

h1, h2, h3, h4, h5, h6 {
    font-weight: 600 !important;
    line-height: 1.2 !important;
    margin-bottom: 0.75rem !important;
}

h1 { font-size: var(--lobster-lab-h1) !important; }
h2 { font-size: var(--lobster-lab-h2) !important; }
h3 { font-size: var(--lobster-lab-h3) !important; }
p, li, span, label {
    font-size: var(--lobster-lab-text) !important;
    line-height: 1.5 !important;
}

/* Allow sidebar text to wrap gracefully */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label {
    white-space: normal !important;
}

/* Harmonise metric widgets so long values do not overflow */
[data-testid="stMetricValue"] {
    font-size: clamp(1.1rem, 1.4vw + 0.6rem, 1.8rem) !important;
    line-height: 1.15 !important;
    word-break: break-word !important;
}

[data-testid="stMetricLabel"] {
    font-size: clamp(0.75rem, 0.8vw + 0.45rem, 1.05rem) !important;
    white-space: normal !important;
    line-height: 1.2 !important;
}

/* Reduce excess spacing inside Streamlit containers */
section[data-testid="stHorizontalBlock"] > div {
    padding-top: 0.25rem !important;
    padding-bottom: 0.25rem !important;
}
</style>
"""


def apply_global_styles() -> None:
    st.markdown(_GLOBAL_STYLE, unsafe_allow_html=True)


LOBSTER_DATASETS = {
    "AMZN": {"levels": [1, 5, 10], "date": "2012-06-21"},
    "AAPL": {"levels": [1, 5, 10, 30, 50], "date": "2012-06-21"},
    "GOOG": {"levels": [1, 5, 10], "date": "2012-06-21"},
    "INTC": {"levels": [1, 5, 10], "date": "2012-06-21"},
    "MSFT": {"levels": [1, 5, 10, 30, 50], "date": "2012-06-21"},
    "SPY": {"levels": [30, 50], "date": "2012-06-21"},
}


def seconds_to_eastern_time(seconds: float, date_str: str = "2012-06-21") -> str:
    eastern = zoneinfo.ZoneInfo("America/New_York")
    base_date = datetime.strptime(date_str, "%Y-%m-%d")
    timestamp = base_date + timedelta(seconds=seconds)
    timestamp_eastern = timestamp.replace(tzinfo=eastern)
    return timestamp_eastern.strftime("%H:%M:%S.%f")[:-3]


def get_dataset_date(ticker_name: str) -> str:
    for ticker, info in LOBSTER_DATASETS.items():
        if ticker in ticker_name:
            return info["date"]
    return "2012-06-21"  # Default fallback


def render_metrics_grid(
    metrics: Iterable[Tuple[str, Any]],
    *,
    columns: int = 3,
) -> None:
    items = list(metrics)
    if not items:
        return

    columns = max(1, columns)

    for start in range(0, len(items), columns):
        row_items = items[start : start + columns]
        row_columns = st.columns(len(row_items))
        for col, (label, value) in zip(row_columns, row_items):
            with col:
                st.metric(label, value)


def discover_local_datasets() -> dict:
    datasets = {}
    cwd = Path.cwd()

    for dir_path in sorted(cwd.glob("LOBSTER_SampleFile_*")):
        if not dir_path.is_dir():
            continue

        parts = dir_path.name.split("_")
        if len(parts) < 5:
            continue

        ticker = parts[2]
        levels = parts[4]

        message_files = list(dir_path.glob("*_message_*.csv"))
        orderbook_files = list(dir_path.glob("*_orderbook_*.csv"))

        if message_files and orderbook_files:
            msg_path = str(message_files[0])
            ob_path = str(orderbook_files[0])

            display_name = f"{ticker} ({levels} levels)"
            datasets[display_name] = (msg_path, ob_path)

    return datasets


def generate_hf_dataset_list() -> dict:
    datasets = {}

    for ticker, info in LOBSTER_DATASETS.items():
        date = info["date"]
        for level in info["levels"]:
            dir_name = f"LOBSTER_SampleFile_{ticker}_{date}_{level}"

            if ticker in ["AAPL", "MSFT"] and level >= 30:
                time_range = "34200000_37800000"
            elif ticker == "SPY":
                time_range = "34200000_37800000"
            else:
                time_range = "34200000_57600000"

            msg_path = f"{dir_name}/{ticker}_{date}_{time_range}_message_{level}.csv"
            ob_path = f"{dir_name}/{ticker}_{date}_{time_range}_orderbook_{level}.csv"

            display_name = f"{ticker} ({level} levels)"
            datasets[display_name] = (msg_path, ob_path)

    return datasets


if USE_HUGGINGFACE:
    ALL_SAMPLE_FILES = generate_hf_dataset_list()
else:
    ALL_SAMPLE_FILES = discover_local_datasets()


@st.cache_data(show_spinner="Loading data from Hugging Face...")
def load_data_from_hf(
    message_path: str, orderbook_path: str, repo_id: str
) -> Tuple[Any, Any]:
    msg_file = hf_hub_download(
        repo_id=repo_id, filename=message_path, repo_type="dataset"
    )
    ob_file = hf_hub_download(
        repo_id=repo_id, filename=orderbook_path, repo_type="dataset"
    )
    # Keep as Polars for better performance
    messages, orderbook = read_lobster(msg_file, ob_file, as_pandas=False)
    # Convert to pandas for compatibility with existing visualization code
    # TODO: Migrate visualization code to use Polars directly for 5-10x performance improvement
    return messages.to_pandas(), orderbook.to_pandas()


@st.cache_data(show_spinner="Loading local data files...")
def load_data(message_path: str, orderbook_path: str) -> Tuple[Any, Any]:
    # Keep as Polars for better performance
    messages, orderbook = read_lobster(message_path, orderbook_path, as_pandas=False)
    # Convert to pandas for compatibility with existing visualization code
    # TODO: Migrate visualization code to use Polars directly for 5-10x performance improvement
    return messages.to_pandas(), orderbook.to_pandas()


@st.cache_data
def precompute_mid_prices(orderbook) -> list:
    """Pre-compute all mid prices for the entire dataset to avoid repeated calculations."""
    mid_prices = []
    for i in range(len(orderbook)):
        ob = orderbook.iloc[i]
        ask_px = ob["ask_price_1"]
        bid_px = ob["bid_price_1"]
        if bid_px > 0 and ask_px < 9999999999:
            mid = (bid_px + ask_px) / 20000.0
            mid_prices.append(mid)
        else:
            mid_prices.append(None)
    return mid_prices


@st.cache_data
def precompute_execution_events(messages) -> dict:
    """Pre-compute execution event indices and statistics."""
    exec_mask = messages["type"].isin([4, 5])
    exec_indices = messages[exec_mask].index.tolist()
    exec_count = len(exec_indices)
    return {
        "indices": exec_indices,
        "count": exec_count,
        "mask": exec_mask,
    }


def validate_ticker_files(ticker_name: str, msg_path: str, ob_path: str) -> bool:
    return Path(msg_path).exists() and Path(ob_path).exists()


def get_available_tickers(sample_files: dict) -> dict:
    if USE_HUGGINGFACE:
        return sample_files
    else:
        available = {}
        for ticker, (msg_path, ob_path) in sample_files.items():
            if validate_ticker_files(ticker, msg_path, ob_path):
                available[ticker] = (msg_path, ob_path)
        return available


def init_session_state():
    if "selected_ticker" not in st.session_state:
        st.session_state.selected_ticker = "MSFT (30 levels)"
    if "current_idx" not in st.session_state:
        st.session_state.current_idx = 0
    if "messages" not in st.session_state:
        st.session_state.messages = None
    if "orderbook" not in st.session_state:
        st.session_state.orderbook = None
    if "data_source" not in st.session_state:
        st.session_state.data_source = "Hugging Face" if USE_HUGGINGFACE else "Local"
    if "data_loaded" not in st.session_state:
        st.session_state.data_loaded = False


def load_ticker_data():
    available_tickers = get_available_tickers(ALL_SAMPLE_FILES)

    if not available_tickers:
        return None, None, available_tickers

    data_source = "Hugging Face" if USE_HUGGINGFACE else "Local Files"
    st.sidebar.info(f"Data source: {data_source}")

    if st.session_state.selected_ticker not in available_tickers:
        st.session_state.selected_ticker = list(available_tickers.keys())[0]

    def _on_ticker_change():
        # Safely get the new ticker value
        new_ticker = st.session_state.get("ticker_selector")
        if new_ticker:
            st.session_state.selected_ticker = new_ticker
        st.session_state.current_idx = 0
        st.session_state.messages = None
        st.session_state.orderbook = None
        st.session_state.order_book = None
        st.session_state.data_loaded = False

    st.sidebar.selectbox(
        "Select Ticker",
        list(available_tickers.keys()),
        index=list(available_tickers.keys()).index(st.session_state.selected_ticker),
        key="ticker_selector",
        on_change=_on_ticker_change,
    )

    selected_ticker = st.session_state.selected_ticker
    msg_path, ob_path = available_tickers[selected_ticker]

    if not st.session_state.data_loaded:
        # Auto-load data lazily when a ticker is selected (no button)
        try:
            with st.spinner(f"Loading LOBSTER data from {data_source}..."):
                if USE_HUGGINGFACE:
                    messages, orderbook = load_data_from_hf(
                        msg_path, ob_path, HF_REPO_ID
                    )
                else:
                    messages, orderbook = load_data(msg_path, ob_path)
                st.session_state.messages = messages
                st.session_state.orderbook = orderbook
                st.session_state.data_loaded = True
                st.rerun()
        except Exception as e:
            st.error(f"Error loading data: {e}")
            if USE_HUGGINGFACE:
                st.error(f"Repository: {HF_REPO_ID}")
            st.error(f"Message path: {msg_path}")
            st.error(f"Orderbook path: {ob_path}")
            return None, None, available_tickers

        return None, None, available_tickers

    messages = st.session_state.messages
    orderbook = st.session_state.orderbook

    if messages is not None and orderbook is not None:
        st.sidebar.success(f"Loaded {len(messages):,} events")

    return messages, orderbook, available_tickers
