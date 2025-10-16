from __future__ import annotations

import os
from collections import OrderedDict
from pathlib import Path
from typing import Tuple

import pandas as pd
import polars as pl
import streamlit as st

from lobster_parsing import read_lobster

from .utils import LOBSTER_DATASETS

HF_REPO_ID = os.getenv("HF_REPO_ID", "totalorganfailure/lobster-data")

if not HF_REPO_ID and hasattr(st, "secrets") and "HF_REPO_ID" in st.secrets:
    HF_REPO_ID = st.secrets["HF_REPO_ID"]

USE_HUGGINGFACE = bool(HF_REPO_ID)

if USE_HUGGINGFACE:
    from huggingface_hub import snapshot_download


def _shrink_polars_frame(df: pl.DataFrame) -> pl.DataFrame:
    """Downcast numeric columns to reduce memory footprint."""

    try:
        return df.with_columns(pl.all().shrink_dtype())
    except Exception:
        return df


def _polars_to_pandas(df: pl.DataFrame) -> pd.DataFrame:
    """Convert a Polars frame to pandas without extension dtypes."""

    return df.to_pandas(use_pyarrow_extension_array=False)


def _dataset_cache_key(repo_or_source: str, message_path: str, orderbook_path: str) -> str:
    return f"{repo_or_source}:{message_path}|{orderbook_path}"


@st.cache_resource(show_spinner="Syncing Hugging Face snapshot...")
def _download_hf_artifacts(
    repo_id: str, message_path: str, orderbook_path: str
) -> Tuple[Path, Path]:
    """Download the requested CSV pair once and reuse from the local cache."""

    snapshot_path = Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            allow_patterns=[message_path, orderbook_path],
        )
    )

    return snapshot_path / message_path, snapshot_path / orderbook_path


@st.cache_resource(show_spinner=False)
def _read_polars_frames(msg_file: str, ob_file: str) -> Tuple[pl.DataFrame, pl.DataFrame]:
    messages_pl, orderbook_pl = read_lobster(msg_file, ob_file, as_pandas=False)
    return _shrink_polars_frame(messages_pl), _shrink_polars_frame(orderbook_pl)


def _remember_polars_frames(messages: pl.DataFrame, orderbook: pl.DataFrame) -> None:
    st.session_state.messages_polars = messages
    st.session_state.orderbook_polars = orderbook


def _ensure_cache_dict(name: str) -> OrderedDict:
    cache = st.session_state.setdefault(name, OrderedDict())
    # Keep only the most recent entries to bound memory usage
    while len(cache) > 32:
        cache.popitem(last=False)
    return cache

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

@st.cache_data(show_spinner="Resolving Hugging Face dataset paths...")
def _resolve_hf_paths(
    message_path: str, orderbook_path: str, repo_id: str
) -> Tuple[str, str]:
    msg_file, ob_file = _download_hf_artifacts(repo_id, message_path, orderbook_path)
    return str(msg_file), str(ob_file)


def _load_polars_frames(message_path: str, orderbook_path: str, repo_label: str) -> None:
    messages_pl, orderbook_pl = _read_polars_frames(message_path, orderbook_path)
    _remember_polars_frames(messages_pl, orderbook_pl)

    st.session_state.data_cache_key = _dataset_cache_key(
        repo_label, message_path, orderbook_path
    )


def _load_pandas_views() -> Tuple[pd.DataFrame, pd.DataFrame]:
    messages_pl = st.session_state.get("messages_polars")
    orderbook_pl = st.session_state.get("orderbook_polars")

    if messages_pl is None or orderbook_pl is None:
        return pd.DataFrame(), pd.DataFrame()

    cache = _ensure_cache_dict("_pandas_frame_cache")
    cache_key = (id(messages_pl), id(orderbook_pl))
    if cache_key in cache:
        cache.move_to_end(cache_key)
        return cache[cache_key]

    messages_pd = _polars_to_pandas(messages_pl)
    orderbook_pd = _polars_to_pandas(orderbook_pl)

    cache[cache_key] = (messages_pd, orderbook_pd)
    return messages_pd, orderbook_pd


def load_data_from_hf(
    message_path: str, orderbook_path: str, repo_id: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    msg_file, ob_file = _resolve_hf_paths(message_path, orderbook_path, repo_id)
    _load_polars_frames(msg_file, ob_file, repo_id)
    return _load_pandas_views()


def load_data(message_path: str, orderbook_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    _load_polars_frames(message_path, orderbook_path, "local")
    return _load_pandas_views()

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


def get_polars_frames() -> Tuple[pl.DataFrame | None, pl.DataFrame | None]:
    return (
        st.session_state.get("messages_polars"),
        st.session_state.get("orderbook_polars"),
    )


def polars_window(
    df: pl.DataFrame | None, start_idx: int, end_idx: int
) -> pl.DataFrame | None:
    if df is None or start_idx >= end_idx:
        return None
    length = end_idx - start_idx
    if start_idx < 0:
        start_idx = 0
    if start_idx >= df.height:
        return None
    length = min(length, df.height - start_idx)
    return df.slice(start_idx, length)


def polars_window_to_pandas(
    df: pl.DataFrame | None, start_idx: int, end_idx: int
) -> pd.DataFrame:
    window = polars_window(df, start_idx, end_idx)
    if window is None:
        return pd.DataFrame()
    return window.to_pandas(use_pyarrow_extension_array=False)

def init_session_state():
    if "selected_ticker" not in st.session_state:
        st.session_state.selected_ticker = "MSFT (30 levels)"
    if "current_idx" not in st.session_state:
        st.session_state.current_idx = 0
    if "messages" not in st.session_state:
        st.session_state.messages = None
    if "orderbook" not in st.session_state:
        st.session_state.orderbook = None
    if "messages_polars" not in st.session_state:
        st.session_state.messages_polars = None
    if "orderbook_polars" not in st.session_state:
        st.session_state.orderbook_polars = None
    if "order_book_last_idx" not in st.session_state:
        st.session_state.order_book_last_idx = None
    if "data_source" not in st.session_state:
        st.session_state.data_source = "Hugging Face" if USE_HUGGINGFACE else "Local"
    if "data_loaded" not in st.session_state:
        st.session_state.data_loaded = False
    if "data_cache_key" not in st.session_state:
        st.session_state.data_cache_key = None
    if "order_book_cache" not in st.session_state:
        st.session_state.order_book_cache = None

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
        st.session_state.messages_polars = None
        st.session_state.orderbook_polars = None
        st.session_state.order_book = None
        st.session_state.order_book_last_idx = None
        st.session_state.data_loaded = False
        st.session_state.data_cache_key = None
        st.session_state.order_book_cache = None

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
                st.session_state.order_book_cache = None
                st.session_state.data_loaded = True
        except Exception as e:
            st.error(f"Error loading data: {e}")
            if USE_HUGGINGFACE:
                st.error(f"Repository: {HF_REPO_ID}")
            st.error(f"Message path: {msg_path}")
            st.error(f"Orderbook path: {ob_path}")
            return None, None, available_tickers

        if st.session_state.data_loaded:
            st.rerun()
        return None, None, available_tickers

    messages = st.session_state.messages
    orderbook = st.session_state.orderbook

    if messages is not None and orderbook is not None:
        st.sidebar.success(f"Loaded {len(messages):,} events")

    return messages, orderbook, available_tickers