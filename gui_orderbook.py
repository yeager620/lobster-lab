import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
from lobster_parsing import read_lobster
from typing import Optional, Tuple


st.set_page_config(
    page_title="LOBSTER Order Book Visualizer",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def load_data(
    message_path: str, orderbook_path: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return read_lobster(message_path, orderbook_path)


def validate_ticker_files(ticker_name: str, msg_path: str, ob_path: str) -> bool:
    return Path(msg_path).exists() and Path(ob_path).exists()


def get_available_tickers(sample_files: dict) -> dict:
    available = {}
    for ticker, (msg_path, ob_path) in sample_files.items():
        if validate_ticker_files(ticker, msg_path, ob_path):
            available[ticker] = (msg_path, ob_path)
    return available


def get_orderbook_depth(
    ob_row: pd.Series, levels: int = 10
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    bids = []
    asks = []

    for i in range(1, levels + 1):
        ask_px_col = f"ask_price_{i}"
        ask_sz_col = f"ask_size_{i}"
        bid_px_col = f"bid_price_{i}"
        bid_sz_col = f"bid_size_{i}"

        if ask_px_col in ob_row and bid_px_col in ob_row:
            ask_px = ob_row[ask_px_col]
            ask_sz = ob_row[ask_sz_col]
            bid_px = ob_row[bid_px_col]
            bid_sz = ob_row[bid_sz_col]

            if ask_px < 9999999999 and ask_sz > 0:
                asks.append({"price": ask_px / 10000.0, "size": ask_sz})
            if bid_px > -9999999999 and bid_sz > 0:
                bids.append({"price": bid_px / 10000.0, "size": bid_sz})

    return pd.DataFrame(bids), pd.DataFrame(asks)


def plot_orderbook(
    bids: pd.DataFrame, asks: pd.DataFrame, current_msg: Optional[pd.Series] = None
) -> go.Figure:
    fig = go.Figure()

    if not bids.empty:
        fig.add_trace(
            go.Bar(
                x=bids["size"],
                y=bids["price"],
                orientation="h",
                name="Bids",
                marker=dict(color="rgba(0, 180, 0, 0.7)"),
                hovertemplate="<b>Bid</b><br>Price: $%{y:.4f}<br>Size: %{x:,}<extra></extra>",
            )
        )

    if not asks.empty:
        fig.add_trace(
            go.Bar(
                x=asks["size"],
                y=asks["price"],
                orientation="h",
                name="Asks",
                marker=dict(color="rgba(255, 50, 50, 0.7)"),
                hovertemplate="<b>Ask</b><br>Price: $%{y:.4f}<br>Size: %{x:,}<extra></extra>",
            )
        )

    if current_msg is not None and current_msg["type"] in [4, 5]:
        exec_price = current_msg["price"] / 10000.0
        exec_size = current_msg["size"]
        direction = "BUY" if current_msg["direction"] == -1 else "SELL"
        fig.add_hline(
            y=exec_price,
            line_dash="dash",
            line_color="yellow",
            line_width=3,
            annotation_text=f"EXECUTION: ${exec_price:.4f} | {exec_size:,} shares | {direction}",
            annotation_position="top right",
            annotation=dict(font=dict(size=14, color="yellow")),
        )

    fig.update_layout(
        title="Order Book Depth",
        xaxis_title="Shares",
        yaxis_title="Price ($)",
        barmode="overlay",
        height=500,
        hovermode="closest",
        showlegend=True,
        margin=dict(l=80, r=80, t=60, b=60),
    )

    return fig


def plot_price_history(
    messages: pd.DataFrame,
    orderbook: pd.DataFrame,
    current_idx: int,
    window_size: int = 1000,
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

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=times,
            y=mid_prices,
            mode="lines",
            name="Mid Price",
            line=dict(color="blue", width=1),
            hovertemplate="Time: %{x:.2f}s<br>Price: $%{y:.4f}<extra></extra>",
        )
    )

    exec_events = msg_window[msg_window["type"].isin([4, 5])]
    if not exec_events.empty:
        exec_prices = exec_events["price"] / 10000.0
        fig.add_trace(
            go.Scatter(
                x=exec_events["time"],
                y=exec_prices,
                mode="markers",
                name="Executions",
                marker=dict(color="orange", size=6, symbol="diamond"),
                hovertemplate="Execution<br>Time: %{x:.2f}s<br>Price: $%{y:.4f}<extra></extra>",
            )
        )

    if current_idx < len(messages):
        current_msg = messages.iloc[current_idx]
        current_ob = orderbook.iloc[current_idx]
        current_mid = (current_ob["ask_price_1"] + current_ob["bid_price_1"]) / 20000.0

        fig.add_trace(
            go.Scatter(
                x=[current_msg["time"]],
                y=[current_mid],
                mode="markers",
                name="Current",
                marker=dict(color="red", size=12, symbol="star"),
                hovertemplate="<b>Current Position</b><br>Time: %{x:.2f}s<br>Price: $%{y:.4f}<extra></extra>",
            )
        )

    fig.update_layout(
        title=f"Price History (Window: {window_size} events)",
        xaxis_title="Time (seconds after midnight)",
        yaxis_title="Price ($)",
        height=350,
        hovermode="x unified",
        showlegend=True,
        margin=dict(l=60, r=60, t=50, b=50),
    )

    return fig


def format_message_details(msg: pd.Series) -> dict:
    message_types = {
        1: "New Order",
        2: "Cancel",
        3: "Delete",
        4: "Exec (Vis)",
        5: "Exec (Hid)",
        7: "Halt",
    }

    direction_map = {1: "Buy", -1: "Sell"}

    return {
        "Time (s)": f"{msg['time']:.2f}",
        "Type": message_types.get(int(msg['type']), 'Unknown'),
        "Order ID": f"{int(msg['order_id']):,}",
        "Size": f"{int(msg['size']):,}",
        "Price": f"${msg['price'] / 10000.0:.2f}",
        "Side": direction_map.get(int(msg["direction"]), "?"),
    }


def main():
    st.title("LOBSTER Order Book Visualizer")

    all_sample_files = {
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
    }

    available_tickers = get_available_tickers(all_sample_files)

    if not available_tickers:
        st.error(
            "No LOBSTER data files found. Please ensure sample data directories exist in the current directory."
        )
        return

    if "selected_ticker" not in st.session_state:
        st.session_state.selected_ticker = list(available_tickers.keys())[0]
    if "current_idx" not in st.session_state:
        st.session_state.current_idx = 0
    if "messages" not in st.session_state:
        st.session_state.messages = None
    if "orderbook" not in st.session_state:
        st.session_state.orderbook = None

    st.sidebar.header("Data Selection")

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
            with st.spinner("Loading LOBSTER data..."):
                messages, orderbook = load_data(msg_path, ob_path)
                st.session_state.messages = messages
                st.session_state.orderbook = orderbook
        else:
            messages = st.session_state.messages
            orderbook = st.session_state.orderbook

        st.sidebar.success(f"Loaded {len(messages):,} events")

        n_levels = len(
            [col for col in orderbook.columns if col.startswith("ask_price_")]
        )

    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.error(f"Message path: {msg_path}")
        st.error(f"Orderbook path: {ob_path}")
        return

    st.sidebar.header("Playback Controls")

    max_idx = len(messages) - 1
    if st.session_state.current_idx > max_idx:
        st.session_state.current_idx = max_idx
    if st.session_state.current_idx < 0:
        st.session_state.current_idx = 0

    step_size = st.sidebar.selectbox(
        "Step Size", [1, 10, 100, 1000], index=1, key="step_size_selector"
    )

    st.sidebar.markdown("**Navigation**")
    nav_col1, nav_col2 = st.sidebar.columns([1, 1])

    with nav_col1:
        if st.button("<<- Back", key="btn_back", use_container_width=True):
            st.session_state.current_idx = max(
                0, st.session_state.current_idx - step_size
            )
            st.rerun()

        if st.button("<- -1", key="btn_back_one", use_container_width=True):
            st.session_state.current_idx = max(0, st.session_state.current_idx - 1)
            st.rerun()

    with nav_col2:
        if st.button("Next ->>", key="btn_next", use_container_width=True):
            st.session_state.current_idx = min(
                max_idx, st.session_state.current_idx + step_size
            )
            st.rerun()

        if st.button("+1 ->", key="btn_next_one", use_container_width=True):
            st.session_state.current_idx = min(
                max_idx, st.session_state.current_idx + 1
            )
            st.rerun()

    if st.sidebar.button("Reset to Start", key="btn_reset", use_container_width=True):
        st.session_state.current_idx = 0
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.subheader("Jump to Event")

    if "last_search_result" not in st.session_state:
        st.session_state.last_search_result = None

    event_type = st.sidebar.selectbox(
        "Event Type",
        [
            (4, "Execution (Visible)"),
            (5, "Execution (Hidden)"),
            (1, "New Order"),
            (2, "Cancel"),
            (3, "Delete"),
        ],
        format_func=lambda x: x[1],
        key="event_type_selector",
    )

    event_col1, event_col2 = st.sidebar.columns(2)
    with event_col1:
        if st.button("<- Prev", key="btn_prev_event", use_container_width=True):
            target_type = event_type[0]
            found_idx = None
            for i in range(st.session_state.current_idx - 1, -1, -1):
                if int(messages.iloc[i]["type"]) == target_type:
                    found_idx = i
                    break

            if found_idx is not None:
                st.session_state.current_idx = found_idx
                st.session_state.last_search_result = (
                    f"Jumped to {event_type[1]} at #{found_idx}"
                )
                st.rerun()
            else:
                st.session_state.last_search_result = (
                    f"No previous {event_type[1]} found"
                )

    with event_col2:
        if st.button("Next ->", key="btn_next_event", use_container_width=True):
            target_type = event_type[0]
            found_idx = None
            for i in range(st.session_state.current_idx + 1, len(messages)):
                if int(messages.iloc[i]["type"]) == target_type:
                    found_idx = i
                    break

            if found_idx is not None:
                st.session_state.current_idx = found_idx
                st.session_state.last_search_result = (
                    f"Jumped to {event_type[1]} at #{found_idx}"
                )
                st.rerun()
            else:
                st.session_state.last_search_result = f"No next {event_type[1]} found"

    if st.session_state.last_search_result:
        if "Jumped" in st.session_state.last_search_result:
            st.sidebar.success(st.session_state.last_search_result)
        else:
            st.sidebar.warning(st.session_state.last_search_result)
        if st.sidebar.button("Clear message", key="clear_search_msg"):
            st.session_state.last_search_result = None
            st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Manual Jump**")

    manual_idx = st.sidebar.number_input(
        "Jump to Message Index",
        min_value=0,
        max_value=max_idx,
        value=st.session_state.current_idx,
        step=1,
        key="manual_idx_input",
    )

    if manual_idx != st.session_state.current_idx:
        st.session_state.current_idx = manual_idx
        st.session_state.last_search_result = None
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.subheader("Visualization Settings")
    display_levels = st.sidebar.slider(
        "Order Book Levels to Display",
        min_value=1,
        max_value=n_levels,
        value=min(10, n_levels),
        step=1,
        key="levels_slider",
    )

    price_window = st.sidebar.slider(
        "Price History Window",
        min_value=100,
        max_value=10000,
        value=1000,
        step=100,
        key="window_slider",
    )

    current_idx = st.session_state.current_idx
    current_msg = messages.iloc[current_idx]
    current_ob = orderbook.iloc[current_idx]

    message_types = {
        1: ("New Limit Order", "blue"),
        2: ("Partial Cancellation", "orange"),
        3: ("Deletion", "red"),
        4: ("Execution (Visible)", "green"),
        5: ("Execution (Hidden)", "green"),
        7: ("Trading Halt", "red"),
    }
    msg_type_text, msg_type_color = message_types.get(
        int(current_msg["type"]), ("Unknown", "gray")
    )

    st.markdown(f"### Event #{current_idx:,} / {len(messages):,}")
    st.markdown(f"**Event Type:** :{msg_type_color}[{msg_type_text}]")

    st.markdown("#### Current Message Details")
    msg_details = format_message_details(current_msg)

    cols = st.columns(6)
    for i, (key, value) in enumerate(msg_details.items()):
        with cols[i]:
            st.metric(label=key, value=value)

    st.markdown("---")

    st.markdown("#### Order Book Statistics")
    best_bid = current_ob["bid_price_1"] / 10000.0
    best_ask = current_ob["ask_price_1"] / 10000.0
    spread = best_ask - best_bid
    mid = (best_bid + best_ask) / 2

    cols = st.columns(4)
    with cols[0]:
        st.metric("Best Bid", f"${best_bid:.4f}")
    with cols[1]:
        st.metric("Best Ask", f"${best_ask:.4f}")
    with cols[2]:
        st.metric("Spread", f"${spread:.4f}")
    with cols[3]:
        st.metric("Mid Price", f"${mid:.4f}")

    st.markdown("---")

    st.markdown("#### Order Book Depth")
    bids, asks = get_orderbook_depth(current_ob, levels=display_levels)

    if not bids.empty and not asks.empty:
        fig_ob = plot_orderbook(bids, asks, current_msg)
        st.plotly_chart(
            fig_ob, use_container_width=True, key=f"orderbook_chart_{current_idx}"
        )
    else:
        st.warning("No valid order book data at this index.")

    st.markdown("---")

    st.markdown("#### Price Time Series")
    fig_price = plot_price_history(
        messages, orderbook, current_idx, window_size=price_window
    )
    st.plotly_chart(
        fig_price, use_container_width=True, key=f"price_chart_{current_idx}"
    )

    with st.expander("View Raw Data"):
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Message Data**")
            st.dataframe(current_msg.to_frame(), use_container_width=True)
        with col2:
            st.write("**Order Book Data (First 10 levels)**")
            ob_display = current_ob.to_frame().head(40)
            st.dataframe(ob_display, use_container_width=True)


if __name__ == "__main__":
    main()
