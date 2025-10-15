"""Shared utilities for GUI pages."""

import streamlit as st
from pathlib import Path
from lobster_parsing import read_lobster
from typing import Tuple, Any
import os

# Configuration: Set HF_REPO_ID environment variable to load from Hugging Face
# Example: export HF_REPO_ID="username/lobster-data"
# Or create a .streamlit/secrets.toml file with: HF_REPO_ID = "username/lobster-data"

# Default to your HuggingFace repository
HF_REPO_ID = os.getenv("HF_REPO_ID", "totalorganfailure/lobster-data")

# If not in environment, try Streamlit secrets
if not HF_REPO_ID and hasattr(st, "secrets") and "HF_REPO_ID" in st.secrets:
    HF_REPO_ID = st.secrets["HF_REPO_ID"]

USE_HUGGINGFACE = bool(HF_REPO_ID)

if USE_HUGGINGFACE:
    from huggingface_hub import hf_hub_download


# LOBSTER dataset configuration matching sync_datasets.py
LOBSTER_DATASETS = {
    "AMZN": {"levels": [1, 5, 10], "date": "2012-06-21"},
    "AAPL": {"levels": [1, 5, 10, 30, 50], "date": "2012-06-21"},
    "GOOG": {"levels": [1, 5, 10], "date": "2012-06-21"},
    "INTC": {"levels": [1, 5, 10], "date": "2012-06-21"},
    "MSFT": {"levels": [1, 5, 10, 30, 50], "date": "2012-06-21"},
    "SPY": {"levels": [30, 50], "date": "2012-06-21"},
}


def discover_local_datasets() -> dict:
    """Discover all LOBSTER dataset directories in the current working directory."""
    datasets = {}
    cwd = Path.cwd()

    # Find all LOBSTER sample directories
    for dir_path in sorted(cwd.glob("LOBSTER_SampleFile_*")):
        if not dir_path.is_dir():
            continue

        # Parse directory name: LOBSTER_SampleFile_TICKER_DATE_LEVELS
        parts = dir_path.name.split("_")
        if len(parts) < 5:
            continue

        ticker = parts[2]
        levels = parts[4]

        # Find message and orderbook CSV files in the directory
        message_files = list(dir_path.glob("*_message_*.csv"))
        orderbook_files = list(dir_path.glob("*_orderbook_*.csv"))

        if message_files and orderbook_files:
            # Use the first matching file
            msg_path = str(message_files[0])
            ob_path = str(orderbook_files[0])

            # Create a display name
            display_name = f"{ticker} ({levels} levels)"
            datasets[display_name] = (msg_path, ob_path)

    return datasets


def generate_hf_dataset_list() -> dict:
    """Generate dataset list for HuggingFace based on LOBSTER_DATASETS configuration."""
    datasets = {}

    for ticker, info in LOBSTER_DATASETS.items():
        date = info["date"]
        for level in info["levels"]:
            # Construct paths matching the HuggingFace repository structure
            dir_name = f"LOBSTER_SampleFile_{ticker}_{date}_{level}"

            # Find the time range - we need to handle different ranges per ticker
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


# Sample file definitions
# Use HuggingFace dataset list if configured, otherwise discover local datasets
if USE_HUGGINGFACE:
    ALL_SAMPLE_FILES = generate_hf_dataset_list()
else:
    ALL_SAMPLE_FILES = discover_local_datasets()


@st.cache_data
def load_data_from_hf(
    message_path: str, orderbook_path: str, repo_id: str
) -> Tuple[Any, Any]:
    """Load and cache LOBSTER data from Hugging Face Hub, converted to pandas for visualization."""
    msg_file = hf_hub_download(
        repo_id=repo_id, filename=message_path, repo_type="dataset"
    )
    ob_file = hf_hub_download(
        repo_id=repo_id, filename=orderbook_path, repo_type="dataset"
    )
    return read_lobster(msg_file, ob_file, as_pandas=True)


@st.cache_data
def load_data(message_path: str, orderbook_path: str) -> Tuple[Any, Any]:
    """Load and cache LOBSTER data from local files, converted to pandas for visualization."""
    return read_lobster(message_path, orderbook_path, as_pandas=True)


def validate_ticker_files(ticker_name: str, msg_path: str, ob_path: str) -> bool:
    """Check if ticker data files exist locally."""
    return Path(msg_path).exists() and Path(ob_path).exists()


def get_available_tickers(sample_files: dict) -> dict:
    """Filter sample_files to only include tickers with valid data."""
    if USE_HUGGINGFACE:
        # When using Hugging Face, all tickers are assumed available
        return sample_files
    else:
        # When using local files, check if they exist
        available = {}
        for ticker, (msg_path, ob_path) in sample_files.items():
            if validate_ticker_files(ticker, msg_path, ob_path):
                available[ticker] = (msg_path, ob_path)
        return available


def init_session_state():
    """Initialize common session state variables."""
    if "selected_ticker" not in st.session_state:
        available = get_available_tickers(ALL_SAMPLE_FILES)
        if available:
            st.session_state.selected_ticker = list(available.keys())[0]
    if "current_idx" not in st.session_state:
        st.session_state.current_idx = 0
    if "messages" not in st.session_state:
        st.session_state.messages = None
    if "orderbook" not in st.session_state:
        st.session_state.orderbook = None
    if "data_source" not in st.session_state:
        st.session_state.data_source = "Hugging Face" if USE_HUGGINGFACE else "Local"


def load_ticker_data():
    """Load the currently selected ticker data. Returns (messages, orderbook, available_tickers) or (None, None, available_tickers) on error."""
    available_tickers = get_available_tickers(ALL_SAMPLE_FILES)

    if not available_tickers:
        return None, None, available_tickers

    # Show data source indicator
    data_source = "Hugging Face" if USE_HUGGINGFACE else "Local Files"
    st.sidebar.info(f"Data source: {data_source}")

    # Handle ticker selection
    selected_ticker = st.sidebar.selectbox(
        "Select Ticker",
        list(available_tickers.keys()),
        index=list(available_tickers.keys()).index(st.session_state.selected_ticker)
        if st.session_state.selected_ticker in available_tickers
        else 0,
        key="ticker_selector",
    )

    # Reset data if ticker changed
    if selected_ticker != st.session_state.selected_ticker:
        st.session_state.selected_ticker = selected_ticker
        st.session_state.current_idx = 0
        st.session_state.messages = None
        st.session_state.orderbook = None

    msg_path, ob_path = available_tickers[selected_ticker]

    # Load data
    try:
        if st.session_state.messages is None or st.session_state.orderbook is None:
            with st.spinner(f"Loading LOBSTER data from {data_source}..."):
                if USE_HUGGINGFACE:
                    messages, orderbook = load_data_from_hf(
                        msg_path, ob_path, HF_REPO_ID
                    )
                else:
                    messages, orderbook = load_data(msg_path, ob_path)
                st.session_state.messages = messages
                st.session_state.orderbook = orderbook
        else:
            messages = st.session_state.messages
            orderbook = st.session_state.orderbook

        st.sidebar.success(f"Loaded {len(messages):,} events")
        return messages, orderbook, available_tickers

    except Exception as e:
        st.error(f"Error loading data: {e}")
        if USE_HUGGINGFACE:
            st.error(f"Repository: {HF_REPO_ID}")
        st.error(f"Message path: {msg_path}")
        st.error(f"Orderbook path: {ob_path}")
        return None, None, available_tickers
