import streamlit as st
from pathlib import Path
from lobster_parsing import read_lobster
from typing import Tuple, Any
import os

HF_REPO_ID = os.getenv("HF_REPO_ID", "totalorganfailure/lobster-data")

if not HF_REPO_ID and hasattr(st, "secrets") and "HF_REPO_ID" in st.secrets:
    HF_REPO_ID = st.secrets["HF_REPO_ID"]

USE_HUGGINGFACE = bool(HF_REPO_ID)

if USE_HUGGINGFACE:
    from huggingface_hub import hf_hub_download


LOBSTER_DATASETS = {
    "AMZN": {"levels": [1, 5, 10], "date": "2012-06-21"},
    "AAPL": {"levels": [1, 5, 10, 30, 50], "date": "2012-06-21"},
    "GOOG": {"levels": [1, 5, 10], "date": "2012-06-21"},
    "INTC": {"levels": [1, 5, 10], "date": "2012-06-21"},
    "MSFT": {"levels": [1, 5, 10, 30, 50], "date": "2012-06-21"},
    "SPY": {"levels": [30, 50], "date": "2012-06-21"},
}


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


@st.cache_data
def load_data_from_hf(
    message_path: str, orderbook_path: str, repo_id: str
) -> Tuple[Any, Any]:
    msg_file = hf_hub_download(
        repo_id=repo_id, filename=message_path, repo_type="dataset"
    )
    ob_file = hf_hub_download(
        repo_id=repo_id, filename=orderbook_path, repo_type="dataset"
    )
    return read_lobster(msg_file, ob_file, as_pandas=True)


@st.cache_data
def load_data(message_path: str, orderbook_path: str) -> Tuple[Any, Any]:
    return read_lobster(message_path, orderbook_path, as_pandas=True)


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
        available = get_available_tickers(ALL_SAMPLE_FILES)
        if available:
            # Default to SPY (30 levels) if available
            default_ticker = "SPY (30 levels)"
            if default_ticker in available:
                st.session_state.selected_ticker = default_ticker
            else:
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
    available_tickers = get_available_tickers(ALL_SAMPLE_FILES)

    if not available_tickers:
        return None, None, available_tickers

    data_source = "Hugging Face" if USE_HUGGINGFACE else "Local Files"
    st.sidebar.info(f"Data source: {data_source}")

    selected_ticker = st.sidebar.selectbox(
        "Select Ticker",
        list(available_tickers.keys()),
        index=list(available_tickers.keys()).index(st.session_state.selected_ticker)
        if st.session_state.selected_ticker in available_tickers
        else 0,
        key="ticker_selector",
    )

    if selected_ticker != st.session_state.selected_ticker:
        st.session_state.selected_ticker = selected_ticker
        st.session_state.current_idx = 0
        st.session_state.messages = None
        st.session_state.orderbook = None

    msg_path, ob_path = available_tickers[selected_ticker]

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
