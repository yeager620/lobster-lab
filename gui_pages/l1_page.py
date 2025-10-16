import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from .shared import (
    init_session_state,
    load_ticker_data,
    seconds_to_eastern_time,
    get_dataset_date,
    apply_global_styles,
    render_metrics_grid,
)


def create_candlestick_data(
    messages: pd.DataFrame, orderbook: pd.DataFrame, window: int = 100
):
    candles = []

    mid_prices = []
    times = []

    for i in range(len(orderbook)):
        ob = orderbook.iloc[i]
        msg = messages.iloc[i]
        ask_px = ob["ask_price_1"]
        bid_px = ob["bid_price_1"]

        if bid_px > 0 and ask_px < 9999999999:
            mid = (bid_px + ask_px) / 20000.0
            mid_prices.append(mid)
            times.append(msg["time"])

    for i in range(0, len(mid_prices) - window, window):
        window_prices = mid_prices[i : i + window]
        window_times = times[i : i + window]

        if len(window_prices) > 0:
            candles.append(
                {
                    "time": window_times[0],
                    "open": window_prices[0],
                    "high": max(window_prices),
                    "low": min(window_prices),
                    "close": window_prices[-1],
                }
            )

    return pd.DataFrame(candles)


def plot_price_candlestick(
    messages: pd.DataFrame,
    orderbook: pd.DataFrame,
    current_idx: int,
    window_size: int = 1000,
    ticker_name: str = "",
) -> go.Figure:
    start_idx = max(0, current_idx - window_size)
    end_idx = min(len(messages), current_idx + window_size // 4)

    msg_window = messages.iloc[start_idx:end_idx]
    ob_window = orderbook.iloc[start_idx:end_idx]

    mid_prices = []
    times = []
    for i in range(len(ob_window)):
        ob = ob_window.iloc[i]
        ask_px = ob["ask_price_1"]
        bid_px = ob["bid_price_1"]
        if bid_px > 0 and ask_px < 9999999999:
            mid = (bid_px + ask_px) / 20000.0
            mid_prices.append(mid)
            times.append(msg_window.iloc[i]["time"])

    candle_window = 50
    candle_data = create_candlestick_data(msg_window, ob_window, window=candle_window)

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


def plot_volume_profile(
    messages: pd.DataFrame,
    orderbook: pd.DataFrame,
    current_idx: int,
    window_size: int = 1000,
) -> go.Figure:
    start_idx = max(0, current_idx - window_size)
    end_idx = min(len(messages), current_idx + 1)

    msg_window = messages.iloc[start_idx:end_idx]

    exec_events = msg_window[msg_window["type"].isin([4, 5])].copy()

    if exec_events.empty:
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

    exec_events["price_level"] = (exec_events["price"] / 10000.0).round(2)

    volume_profile = exec_events.groupby("price_level")["size"].sum().reset_index()
    volume_profile.columns = ["price", "volume"]
    volume_profile = volume_profile.sort_values("price")

    exec_events_buy = exec_events[exec_events["direction"] == 1]
    exec_events_sell = exec_events[exec_events["direction"] == -1]

    buy_profile = exec_events_buy.groupby("price_level")["size"].sum().reset_index()
    sell_profile = exec_events_sell.groupby("price_level")["size"].sum().reset_index()

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
    st.sidebar.subheader("Time Navigation")

    max_idx = len(messages) - 1

    current_idx = st.sidebar.slider(
        "Event Index",
        min_value=0,
        max_value=max_idx,
        value=st.session_state.current_idx,
        step=1,
        key="l1_idx_slider",
    )
    st.session_state.current_idx = current_idx

    jump_col1, jump_col2, jump_col3 = st.sidebar.columns(3)
    with jump_col1:
        if st.button("-1000", key="l1_jump_back_1000", use_container_width=True):
            st.session_state.current_idx = max(0, current_idx - 1000)
            st.rerun()
    with jump_col2:
        if st.button("Reset", key="l1_jump_reset", use_container_width=True):
            st.session_state.current_idx = 0
            st.rerun()
    with jump_col3:
        if st.button("+1000", key="l1_jump_fwd_1000", use_container_width=True):
            st.session_state.current_idx = min(max_idx, current_idx + 1000)
            st.rerun()

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
        fig_candle, use_container_width=True, key=f"l1_candle_{current_idx}"
    )

    st.markdown("---")

    st.markdown("### Volume Profile")
    st.markdown("*Distribution of executed volume by price level*")
    fig_volume = plot_volume_profile(
        messages, orderbook, current_idx, window_size=chart_window
    )
    st.plotly_chart(
        fig_volume, use_container_width=True, key=f"l1_volume_{current_idx}"
    )
