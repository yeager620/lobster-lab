"""Shared utilities for GUI pages."""

import streamlit as st
from pathlib import Path
from lobster_parsing import read_lobster
from typing import Tuple, Any
import os

# Configuration: Set HF_REPO_ID environment variable to load from Hugging Face
# Example: export HF_REPO_ID="username/lobster-data"
# Or create a .streamlit/secrets.toml file with: HF_REPO_ID = "username/lobster-data"

# Try to get from environment variable first
HF_REPO_ID = os.getenv("HF_REPO_ID", "")

# If not in environment, try Streamlit secrets
if not HF_REPO_ID and hasattr(st, "secrets") and "HF_REPO_ID" in st.secrets:
    HF_REPO_ID = st.secrets["HF_REPO_ID"]

USE_HUGGINGFACE = bool(HF_REPO_ID)

if USE_HUGGINGFACE:
    from huggingface_hub import hf_hub_download


# Sample file definitions
# Note: Add directories to match your local files or HF repository structure
# Sample datasets available at: https://lobsterdata.com/info/DataSamples.php
ALL_SAMPLE_FILES = {
    "AAPL (50 levels)": (
        "LOBSTER_SampleFile_AAPL_2012-06-21_50/AAPL_2012-06-21_34200000_37800000_message_50.csv",
        "LOBSTER_SampleFile_AAPL_2012-06-21_50/AAPL_2012-06-21_34200000_37800000_orderbook_50.csv",
    ),
    "MSFT (50 levels)": (
        "LOBSTER_SampleFile_MSFT_2012-06-21_50/MSFT_2012-06-21_34200000_37800000_message_50.csv",
        "LOBSTER_SampleFile_MSFT_2012-06-21_50/MSFT_2012-06-21_34200000_37800000_orderbook_50.csv",
    ),
    "SPY (50 levels)": (
        "LOBSTER_SampleFile_SPY_2012-06-21_50/SPY_2012-06-21_34200000_37800000_message_50.csv",
        "LOBSTER_SampleFile_SPY_2012-06-21_50/SPY_2012-06-21_34200000_37800000_orderbook_50.csv",
    ),
    "GOOG (10 levels)": (
        "LOBSTER_SampleFile_GOOG_2012-06-21_10/GOOG_2012-06-21_34200000_57600000_message_10.csv",
        "LOBSTER_SampleFile_GOOG_2012-06-21_10/GOOG_2012-06-21_34200000_57600000_orderbook_10.csv",
    ),
    "INTC (10 levels)": (
        "LOBSTER_SampleFile_INTC_2012-06-21_10/INTC_2012-06-21_34200000_57600000_message_10.csv",
        "LOBSTER_SampleFile_INTC_2012-06-21_10/INTC_2012-06-21_34200000_57600000_orderbook_10.csv",
    ),
}


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
