from collections import OrderedDict

import polars as pl
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .data import (
    init_session_state,
    load_ticker_data,
    get_polars_frames,
    polars_window_to_pandas,
)
from .utils import (
    seconds_to_eastern_time,
    get_dataset_date,
)
from .ui import (
    apply_global_styles,
    render_metrics_grid,
    render_sidebar,
)


def _get_cache(name: str, max_entries: int = 64) -> OrderedDict:
    cache = st.session_state.setdefault(name, OrderedDict())
    while len(cache) > max_entries:
        cache.popitem(last=False)
    return cache


@st.cache_data
def _create_candlestick_data_pandas(
    messages: pd.DataFrame, orderbook: pd.DataFrame, window: int = 100
) -> pd.DataFrame:

    if messages.empty or orderbook.empty:
        return pd.DataFrame(columns=["time", "open", "high", "low", "close"])

    df = orderbook[["bid_price_1", "ask_price_1"]].copy()
    df["time"] = messages["time"].to_numpy()

    valid = (df["bid_price_1"] > 0) & (df["ask_price_1"] < 9_999_999_999)
    df = df.loc[valid].copy()

    if df.empty:
        return pd.DataFrame(columns=["time", "open", "high", "low", "close"])

    df["mid"] = (df["bid_price_1"] + df["ask_price_1"]) / 20_000.0
    df.reset_index(drop=True, inplace=True)
    df["bucket"] = df.index // max(window, 1)

    grouped = df.groupby("bucket", sort=True)
    candles = grouped.agg(
        time=("time", "first"),
        open=("mid", "first"),
        high=("mid", "max"),
        low=("mid", "min"),
        close=("mid", "last"),
    )

    return candles.reset_index(drop=True)


def _create_candlestick_data_polars(
    messages_pl: pl.DataFrame,
    orderbook_pl: pl.DataFrame,
    start_idx: int,
    end_idx: int,
    window: int,
) -> pd.DataFrame | None:
    if start_idx >= end_idx:
        return None

    length = end_idx - start_idx
    msg_window = messages_pl.slice(start_idx, length)
    ob_window = orderbook_pl.slice(start_idx, length).select(
        ["bid_price_1", "ask_price_1"]
    )

    if ob_window.height == 0:
        return None

    bucket = max(window, 1)
    enriched = (
        ob_window.with_columns(
            time=msg_window["time"],
            mid=(pl.col("bid_price_1") + pl.col("ask_price_1")) / 20000.0,
            bucket=(pl.arange(0, ob_window.height) // bucket),
        )
        .filter(
            (pl.col("bid_price_1") > 0)
            & (pl.col("ask_price_1") < 9_999_999_999)
        )
    )

    if enriched.height == 0:
        return None

    candles = (
        enriched.group_by("bucket")
        .agg(
            time=pl.col("time").first(),
            open=pl.col("mid").first(),
            high=pl.col("mid").max(),
            low=pl.col("mid").min(),
            close=pl.col("mid").last(),
        )
        .sort("time")
        .select(["time", "open", "high", "low", "close"])
    )

    return candles.to_pandas(use_pyarrow_extension_array=False)


def create_candlestick_data(
    messages_window: pd.DataFrame,
    orderbook_window: pd.DataFrame,
    start_idx: int,
    end_idx: int,
    window: int = 100,
) -> pd.DataFrame:
    messages_pl, orderbook_pl = get_polars_frames()
    data_cache_key = st.session_state.get("data_cache_key")
    cache = _get_cache("candlestick_cache")
    cache_key = (data_cache_key, start_idx, end_idx, window)

    if cache_key in cache:
        cache.move_to_end(cache_key)
        return cache[cache_key]

    if messages_pl is not None and orderbook_pl is not None:
        candles = _create_candlestick_data_polars(
            messages_pl, orderbook_pl, start_idx, end_idx, window
        )
        if candles is not None:
            cache[cache_key] = candles
            return candles

    candles = _create_candlestick_data_pandas(
        messages=messages_window, orderbook=orderbook_window, window=window
    )
    cache[cache_key] = candles
    return candles

def plot_price_candlestick(
    messages: pd.DataFrame,
    orderbook: pd.DataFrame,
    current_idx: int,
    window_size: int = 1000,
    ticker_name: str = "",
) -> go.Figure:
    start_idx = max(0, current_idx - window_size)
    end_idx = min(len(messages), current_idx + window_size // 4)

    messages_pl, orderbook_pl = get_polars_frames()
    if messages_pl is not None:
        msg_window = polars_window_to_pandas(messages_pl, start_idx, end_idx)
    else:
        msg_window = messages.iloc[start_idx:end_idx]

    if orderbook_pl is not None:
        ob_window = polars_window_to_pandas(orderbook_pl, start_idx, end_idx)
    else:
        ob_window = orderbook.iloc[start_idx:end_idx]

    candle_window = 50
    candle_data = create_candlestick_data(
        msg_window,
        ob_window,
        start_idx,
        end_idx,
        window=candle_window,
    )

    date_str = get_dataset_date(ticker_name)

    fig = go.Figure()

    if not candle_data.empty:
        candle_times_et = [
            seconds_to_eastern_time(t, date_str) for t in candle_data["time"]
        ]
        fig.add_trace(
            go.Candlestick(
                x=candle_times_et,
                open=candle_data["open"],
                high=candle_data["high"],
                low=candle_data["low"],
                close=candle_data["close"],
                name="Price",
                increasing_line_color="green",
                decreasing_line_color="red",
            )
        )

    exec_events = msg_window[msg_window["type"].isin([4, 5])]
    if not exec_events.empty:
        exec_prices = exec_events["price"].astype(float) / 10000.0
        exec_times_et = [
            seconds_to_eastern_time(t, date_str) for t in exec_events["time"]
        ]
        fig.add_trace(
            go.Scatter(
                x=exec_times_et,
                y=exec_prices.astype(float),
                mode="markers",
                name="Executions",
                marker=dict(color="orange", size=6, symbol="diamond"),
                hovertemplate="Execution<br>Time: %{x}<br>Price: $%{y:.4f}<extra></extra>",
            )
        )

    if current_idx < len(messages):
        current_msg = messages.iloc[current_idx]
        current_ob = orderbook.iloc[current_idx]
        current_mid = (current_ob["ask_price_1"] + current_ob["bid_price_1"]) / 20000.0
        current_time_et = seconds_to_eastern_time(current_msg["time"], date_str)

        fig.add_trace(
            go.Scatter(
                x=[current_time_et],
                y=[current_mid],
                mode="markers",
                name="Current",
                marker=dict(
                    color="cyan",
                    size=15,
                    symbol="star",
                    line=dict(width=2, color="white"),
                ),
                hovertemplate="<b>Current Position</b><br>Time: %{x}<br>Price: $%{y:.4f}<extra></extra>",
            )
        )

    fig.update_layout(
        title=f"Price Candlestick Chart (Window: {window_size} events)",
        xaxis_title="Time (Eastern)",
        yaxis_title="Price ($)",
        height=500,
        hovermode="x unified",
        showlegend=True,
        margin=dict(l=60, r=60, t=50, b=50),
        xaxis_rangeslider_visible=False,
    )

    return fig


@st.cache_data
def _compute_volume_profile_pandas(
    messages: pd.DataFrame, start_idx: int, end_idx: int
) -> tuple:
    msg_window = messages.iloc[start_idx:end_idx]
    exec_events = msg_window[msg_window["type"].isin([4, 5])].copy()

    if exec_events.empty:
        return None, None, None

    exec_events["price_level"] = (exec_events["price"] / 10000.0).round(2)

    exec_events_buy = exec_events[exec_events["direction"] == 1]
    exec_events_sell = exec_events[exec_events["direction"] == -1]

    buy_profile = exec_events_buy.groupby("price_level")["size"].sum().reset_index()
    sell_profile = exec_events_sell.groupby("price_level")["size"].sum().reset_index()

    return exec_events, buy_profile, sell_profile


def _compute_volume_profile_polars(
    messages_pl: pl.DataFrame, start_idx: int, end_idx: int
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None]:
    if start_idx >= end_idx:
        return None, None, None

    length = end_idx - start_idx
    msg_window = messages_pl.slice(start_idx, length)
    exec_events = msg_window.filter(pl.col("type").is_in([4, 5]))

    if exec_events.height == 0:
        return None, None, None

    enriched = exec_events.with_columns(
        price_level=(pl.col("price") / 10000.0).round(2),
        price=pl.col("price") / 10000.0,
    )

    buy_profile = (
        enriched.filter(pl.col("direction") == 1)
        .groupby("price_level")
        .agg(size=pl.col("size").sum())
        .sort("price_level")
        .to_pandas(use_pyarrow_extension_array=False)
    )

    sell_profile = (
        enriched.filter(pl.col("direction") == -1)
        .groupby("price_level")
        .agg(size=pl.col("size").sum())
        .sort("price_level")
        .to_pandas(use_pyarrow_extension_array=False)
    )

    exec_pd = enriched.to_pandas(use_pyarrow_extension_array=False)
    return exec_pd, buy_profile, sell_profile


def compute_volume_profile(
    messages: pd.DataFrame, start_idx: int, end_idx: int
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None]:
    messages_pl, _ = get_polars_frames()
    data_cache_key = st.session_state.get("data_cache_key")
    cache = _get_cache("volume_profile_cache")
    cache_key = (data_cache_key, start_idx, end_idx)

    if cache_key in cache:
        cache.move_to_end(cache_key)
        return cache[cache_key]

    if messages_pl is not None:
        exec_events, buy_profile, sell_profile = _compute_volume_profile_polars(
            messages_pl, start_idx, end_idx
        )
        if exec_events is not None:
            cache[cache_key] = (exec_events, buy_profile, sell_profile)
            return exec_events, buy_profile, sell_profile

    exec_events, buy_profile, sell_profile = _compute_volume_profile_pandas(
        messages, start_idx, end_idx
    )
    cache[cache_key] = (exec_events, buy_profile, sell_profile)
    return exec_events, buy_profile, sell_profile


def plot_volume_profile(
    messages: pd.DataFrame,
    orderbook: pd.DataFrame,
    current_idx: int,
    window_size: int = 1000,
) -> go.Figure:
    start_idx = max(0, current_idx - window_size)
    end_idx = min(len(messages), current_idx + 1)

    exec_events, buy_profile, sell_profile = compute_volume_profile(
        messages, start_idx, end_idx
    )

    if exec_events is None:
        fig = go.Figure()
        fig.add_annotation(
            text="No executions in current window",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return fig

    fig = go.Figure()

    if not buy_profile.empty:
        fig.add_trace(
            go.Bar(
                x=buy_profile.iloc[:, 1],
                y=buy_profile.iloc[:, 0],
                orientation="h",
                name="Buy Volume",
                marker_color="green",
                opacity=0.7,
                hovertemplate="Price: $%{y:.2f}<br>Volume: %{x:,}<extra></extra>",
            )
        )

    if not sell_profile.empty:
        fig.add_trace(
            go.Bar(
                x=-sell_profile.iloc[:, 1],
                y=sell_profile.iloc[:, 0],
                orientation="h",
                name="Sell Volume",
                marker_color="red",
                opacity=0.7,
                hovertemplate="Price: $%{y:.2f}<br>Volume: %{x:,}<extra></extra>",
            )
        )

    if current_idx < len(messages):
        current_ob = orderbook.iloc[current_idx]
        current_mid = (current_ob["ask_price_1"] + current_ob["bid_price_1"]) / 20000.0

        fig.add_hline(
            y=current_mid,
            line_dash="dash",
            line_color="cyan",
            line_width=2,
            annotation_text=f"Current: ${current_mid:.2f}",
            annotation_position="right",
        )

    fig.update_layout(
        title=f"Volume Profile (Last {window_size} events)",
        xaxis_title="Volume (shares)",
        yaxis_title="Price ($)",
        height=500,
        barmode="overlay",
        showlegend=True,
        margin=dict(l=60, r=80, t=50, b=50),
    )

    return fig


def show():
    apply_global_styles()
    st.title("L1")
    st.markdown("**Price action, candlesticks, and volume analysis**")

    init_session_state()

    messages, orderbook, available_tickers = load_ticker_data()

    if messages is None:
        st.error(
            "No LOBSTER data files found. Please ensure sample data directories exist."
        )
        return

    render_sidebar(playback_enabled=False)

    st.sidebar.markdown("---")
    st.sidebar.header("Visualization Settings")

    chart_window = st.sidebar.slider(
        "Chart Window Size",
        min_value=100,
        max_value=5000,
        value=1000,
        step=100,
        key="l1_window_slider",
    )

    st.sidebar.markdown("---")

    current_idx = st.session_state.current_idx

    current_ob = orderbook.iloc[current_idx]

    st.markdown(f"### Event #{current_idx:,} / {len(messages):,}")

    best_bid = current_ob["bid_price_1"] / 10000.0
    best_ask = current_ob["ask_price_1"] / 10000.0
    spread = best_ask - best_bid
    mid = (best_bid + best_ask) / 2

    render_metrics_grid(
        [
            ("Best Bid", f"${best_bid:.4f}"),
            ("Best Ask", f"${best_ask:.4f}"),
            ("Spread", f"${spread:.4f}"),
            ("Mid Price", f"${mid:.4f}"),
        ],
        columns=2,
    )

    st.markdown("---")

    st.markdown("### Price Candlestick Chart")
    fig_candle = plot_price_candlestick(
        messages,
        orderbook,
        current_idx,
        window_size=chart_window,
        ticker_name=st.session_state.selected_ticker,
    )
    st.plotly_chart(
        fig_candle, width="stretch", key=f"l1_candle_{current_idx}", config={}
    )

    st.markdown("---")

    st.markdown("### Volume Profile")
    st.markdown("*Distribution of executed volume by price level*")
    fig_volume = plot_volume_profile(
        messages, orderbook, current_idx, window_size=chart_window
    )
    st.plotly_chart(
        fig_volume, width="stretch", key=f"l1_volume_{current_idx}", config={}
    )
