from collections import defaultdict
from typing import Dict, List, Tuple

import pandas as pd
import plotly.colors as pc
import plotly.graph_objects as go
import streamlit as st
from .data import (
    init_session_state,
    load_ticker_data,
)
from .utils import (
    seconds_to_eastern_time,
    get_dataset_date,
)
from .ui import (
    apply_global_styles,
    render_metrics_grid,
    render_sidebar as render_shared_sidebar,
)


def update_order_book(order_book: Dict, msg: pd.Series) -> Dict:
    msg_type = int(msg["type"])
    order_id = int(msg["order_id"])
    size = int(msg["size"])
    price = msg["price"] / 10000.0
    direction = int(msg["direction"])
    time = msg["time"]

    side = "bids" if direction == 1 else "asks"

    if msg_type == 1:
        order_book[side][order_id] = {
            "price": price,
            "size": size,
            "time": time,
            "initial_size": size,
        }
    elif msg_type == 2:
        if order_id in order_book[side]:
            order_book[side][order_id]["size"] -= size
            if order_book[side][order_id]["size"] <= 0:
                del order_book[side][order_id]
    elif msg_type == 3:
        if order_id in order_book[side]:
            del order_book[side][order_id]
    elif msg_type in [4, 5]:
        if order_id in order_book[side]:
            order_book[side][order_id]["size"] -= size
            if order_book[side][order_id]["size"] <= 0:
                del order_book[side][order_id]

    return order_book
class OrderBookReplayCache:
    def __init__(self) -> None:
        self.cache_key: str | None = None
        self.max_lookback: int = 50000
        self._order_book: Dict = {"bids": {}, "asks": {}}
        self._last_idx: int = -1

    def reset(self) -> None:
        self._order_book = {"bids": {}, "asks": {}}
        self._last_idx = -1

    def _clone_state(self) -> Dict:
        return {
            "bids": {k: v.copy() for k, v in self._order_book["bids"].items()},
            "asks": {k: v.copy() for k, v in self._order_book["asks"].items()},
        }

    def get_state(
        self,
        cache_key: str | None,
        messages: pd.DataFrame,
        up_to_idx: int,
        max_lookback: int,
    ) -> Dict:
        if cache_key != self.cache_key or max_lookback != self.max_lookback:
            self.cache_key = cache_key
            self.max_lookback = max_lookback
            self.reset()

        if up_to_idx < 0:
            return {"bids": {}, "asks": {}}

        target_start = max(0, up_to_idx - self.max_lookback)

        needs_rebuild = (
            self._last_idx < target_start
            or up_to_idx < self._last_idx
            or self._last_idx == -1
        )

        if needs_rebuild:
            self._order_book = {"bids": {}, "asks": {}}
            replay_start = target_start
        else:
            replay_start = self._last_idx + 1

        replay_end = min(up_to_idx, len(messages) - 1)

        if replay_start <= replay_end:
            for idx in range(replay_start, replay_end + 1):
                msg = messages.iloc[idx]
                update_order_book(self._order_book, msg)

        self._last_idx = max(self._last_idx, replay_end)
        return self._clone_state()


def _get_replay_cache() -> OrderBookReplayCache:
    cache = st.session_state.get("order_book_cache")
    if not isinstance(cache, OrderBookReplayCache):
        cache = OrderBookReplayCache()
        st.session_state.order_book_cache = cache
    return cache


def reconstruct_order_book_state(
    messages: pd.DataFrame, up_to_idx: int, max_lookback: int = 50000
) -> Dict:
    cache = _get_replay_cache()
    data_cache_key = st.session_state.get("data_cache_key")
    return cache.get_state(data_cache_key, messages, up_to_idx, max_lookback)


def _sample_queue_color(
    colorscale: str, queue_idx: int, queue_length: int, fallback: str
) -> str:
    if queue_length <= 1:
        position = 0.0
    else:
        denominator = max(queue_length - 1, 1)
        position = min(max(queue_idx / denominator, 0.0), 1.0)

    try:
        sampled = pc.sample_colorscale(colorscale, [position])
        if sampled:
            return sampled[0]
    except (ValueError, TypeError, IndexError):
        pass

    try:
        resolved_scale = pc.get_colorscale(colorscale)
        if resolved_scale:
            return resolved_scale[-1][1]
    except (ValueError, TypeError, IndexError):
        pass

    return fallback


@st.cache_data
def format_order_queue_table(
    price_levels: List, side: str, date_str: str = "2012-06-21"
) -> pd.DataFrame:
    rows = []
    for price, orders in price_levels:
        orders_sorted = sorted(orders, key=lambda x: x["time"])
        for i, order in enumerate(orders_sorted):
            rows.append(
                {
                    "Price": f"${price:.2f}",
                    "Queue Pos": f"{i + 1}/{len(orders_sorted)}",
                    "Order ID": order["id"],
                    "Size": f"{order['size']:,}",
                    "Time": seconds_to_eastern_time(order["time"], date_str),
                }
            )

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def plot_order_book_depth_with_queue(
    order_book: Dict, levels: int = 10
) -> Tuple[go.Figure, List, List]:
    bid_prices = defaultdict(list)
    for order_id, order in order_book["bids"].items():
        bid_prices[order["price"]].append(
            {"id": order_id, "size": order["size"], "time": order["time"]}
        )

    ask_prices = defaultdict(list)
    for order_id, order in order_book["asks"].items():
        ask_prices[order["price"]].append(
            {"id": order_id, "size": order["size"], "time": order["time"]}
        )

    bid_levels = sorted(bid_prices.items(), key=lambda x: x[0], reverse=True)[:levels]
    ask_levels = sorted(ask_prices.items(), key=lambda x: x[0])[:levels]

    fig = go.Figure()

    if bid_levels:
        for idx, (price, orders) in enumerate(bid_levels):
            orders_sorted = sorted(orders, key=lambda x: x["time"])
            total_size = sum(o["size"] for o in orders_sorted)
            queue_length = len(orders_sorted)

            hover_lines = [
                f"<b>Bid @ ${float(price):.2f}</b>",
                f"Total Size: {total_size:,} shares",
                f"Orders in Queue: {queue_length}",
                "<br><b>Queue Details:</b>",
            ]

            for i, order in enumerate(orders_sorted[:5]):
                hover_lines.append(
                    f"  #{i + 1}: {order['size']:,} sh (ID: {order['id']})"
                )
            if len(orders_sorted) > 5:
                hover_lines.append(f"  ... and {len(orders_sorted) - 5} more")

            fig.add_trace(
                go.Bar(
                    x=[total_size],
                    y=[float(price)],
                    orientation="h",
                    name="Bids",
                    marker=dict(color="rgba(0, 180, 0, 0.7)"),
                    hovertemplate="%{customdata}<extra></extra>",
                    customdata=["<br>".join(hover_lines)],
                    showlegend=True if idx == 0 else False,
                    legendgroup="bids",
                    width=[0.004],
                )
            )

            cumulative = 0
            for order in orders_sorted:
                cumulative += order["size"]
                if cumulative < total_size:
                    fig.add_shape(
                        type="line",
                        x0=cumulative,
                        x1=cumulative,
                        y0=float(price) - 0.003,
                        y1=float(price) + 0.003,
                        line=dict(color="rgba(255, 255, 255, 0.6)", width=2),
                        layer="above",
                    )

    if ask_levels:
        for idx, (price, orders) in enumerate(ask_levels):
            orders_sorted = sorted(orders, key=lambda x: x["time"])
            total_size = sum(o["size"] for o in orders_sorted)
            queue_length = len(orders_sorted)

            hover_lines = [
                f"<b>Ask @ ${float(price):.2f}</b>",
                f"Total Size: {total_size:,} shares",
                f"Orders in Queue: {queue_length}",
                "<br><b>Queue Details:</b>",
            ]

            for i, order in enumerate(orders_sorted[:5]):
                hover_lines.append(
                    f"  #{i + 1}: {order['size']:,} sh (ID: {order['id']})"
                )
            if len(orders_sorted) > 5:
                hover_lines.append(f"  ... and {len(orders_sorted) - 5} more")

            fig.add_trace(
                go.Bar(
                    x=[total_size],
                    y=[float(price)],
                    orientation="h",
                    name="Asks",
                    marker=dict(color="rgba(255, 50, 50, 0.7)"),
                    hovertemplate="%{customdata}<extra></extra>",
                    customdata=["<br>".join(hover_lines)],
                    showlegend=True if idx == 0 else False,
                    legendgroup="asks",
                    width=[0.004],
                )
            )

            cumulative = 0
            for order in orders_sorted:
                cumulative += order["size"]
                if cumulative < total_size:
                    fig.add_shape(
                        type="line",
                        x0=cumulative,
                        x1=cumulative,
                        y0=float(price) - 0.003,
                        y1=float(price) + 0.003,
                        line=dict(color="rgba(255, 255, 255, 0.6)", width=2),
                        layer="above",
                    )

    max_size = 0
    if bid_levels:
        max_size = max(
            max_size, max(sum(o["size"] for o in orders) for _, orders in bid_levels)
        )
    if ask_levels:
        max_size = max(
            max_size, max(sum(o["size"] for o in orders) for _, orders in ask_levels)
        )

    fig.update_layout(
        title="Order Book Depth",
        xaxis_title="Shares",
        yaxis_title="Price ($)",
        barmode="overlay",
        bargap=0.2,
        bargroupgap=0.1,
        height=500,
        hovermode="closest",
        showlegend=True,
        margin=dict(l=80, r=80, t=60, b=60),
    )

    return fig, bid_levels, ask_levels


def plot_order_size_distribution(order_book: Dict) -> go.Figure:
    bid_sizes = [order["size"] for order in order_book["bids"].values()]
    ask_sizes = [order["size"] for order in order_book["asks"].values()]

    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=bid_sizes, name="Bid Orders", marker_color="green", opacity=0.7, nbinsx=30
        )
    )

    fig.add_trace(
        go.Histogram(
            x=ask_sizes, name="Ask Orders", marker_color="red", opacity=0.7, nbinsx=30
        )
    )

    fig.update_layout(
        title="Order Size Distribution",
        xaxis_title="Order Size (shares)",
        yaxis_title="Count",
        barmode="overlay",
        height=400,
    )

    return fig


@st.cache_data
def plot_order_timeline(
    messages: pd.DataFrame, order_id: int, date_str: str = "2012-06-21"
) -> go.Figure:
    order_msgs = messages[messages["order_id"] == order_id].copy()

    if order_msgs.empty:
        fig = go.Figure()
        fig.add_annotation(
            text=f"Order ID {order_id} not found in message data",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return fig

    order_msgs = order_msgs.sort_values("time")

    event_map = {
        1: "Submitted",
        2: "Partial Cancel",
        3: "Deleted",
        4: "Executed (Vis)",
        5: "Executed (Hid)",
    }

    events = [event_map.get(int(t), "Unknown") for t in order_msgs["type"]]
    times = order_msgs["time"].values
    times_et = [seconds_to_eastern_time(t, date_str) for t in times]
    sizes = order_msgs["size"].values
    prices = order_msgs["price"].values / 10000.0

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=times_et,
            y=sizes,
            mode="lines+markers",
            marker=dict(size=12, color="blue"),
            line=dict(color="blue", width=2),
            text=[
                f"Event: {e}<br>Time: {t}<br>Size: {s}<br>Price: ${p:.2f}"
                for e, t, s, p in zip(events, times_et, sizes, prices)
            ],
            hovertemplate="%{text}<extra></extra>",
            name=f"Order {order_id}",
        )
    )

    fig.update_layout(
        title=f"Order Lifecycle: ID {order_id}",
        xaxis_title="Time (Eastern)",
        yaxis_title="Remaining Size",
        height=400,
    )

    return fig


@st.cache_data
def plot_order_flow_rate(
    messages: pd.DataFrame, window: int = 100, date_str: str = "2012-06-21"
) -> go.Figure:
    new_orders = messages[messages["type"] == 1].copy()

    if len(new_orders) < window:
        fig = go.Figure()
        fig.add_annotation(
            text="Not enough data for order flow analysis",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
        )
        return fig

    times = []
    times_et = []
    rates = []

    for i in range(window, len(new_orders)):
        time_window = new_orders.iloc[i - window : i]
        time_diff = time_window["time"].iloc[-1] - time_window["time"].iloc[0]
        if time_diff > 0:
            rate = window / time_diff
            time_val = time_window["time"].iloc[-1]
            times.append(time_val)
            times_et.append(seconds_to_eastern_time(time_val, date_str))
            rates.append(rate)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=times_et,
            y=rates,
            mode="lines",
            line=dict(color="purple", width=2),
            name="Order Arrival Rate",
        )
    )

    fig.update_layout(
        title=f"Order Arrival Rate (Rolling Window: {window} orders)",
        xaxis_title="Time (Eastern)",
        yaxis_title="Orders per Second",
        height=400,
    )

    return fig


def render_l3_sidebar() -> int:
    render_shared_sidebar(playback_enabled=True)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Visualization Settings")
    display_levels = st.sidebar.slider(
        "Order Book Levels to Display",
        min_value=1,
        max_value=20,
        value=10,
        step=1,
        key="l3_levels_slider",
    )

    st.sidebar.markdown("---")
    return display_levels


def _ensure_order_book_state(messages: pd.DataFrame, target_idx: int) -> Dict:
    if "order_book_last_idx" not in st.session_state:
        st.session_state.order_book_last_idx = None

    last_idx = st.session_state.order_book_last_idx
    order_book = st.session_state.order_book

    if order_book is None or last_idx is None:
        order_book = reconstruct_order_book_state(messages, target_idx)
        st.session_state.order_book = order_book
        st.session_state.order_book_last_idx = target_idx
        return order_book

    if target_idx < last_idx:
        order_book = reconstruct_order_book_state(messages, target_idx)
        st.session_state.order_book = order_book
        st.session_state.order_book_last_idx = target_idx
        return order_book

    if target_idx > last_idx:
        for idx in range(last_idx + 1, target_idx + 1):
            msg = messages.iloc[idx]
            order_book = update_order_book(order_book, msg)
        st.session_state.order_book = order_book
        st.session_state.order_book_last_idx = target_idx

    return order_book


def render_main_content(state, display_levels):
    current_msg = st.session_state.messages.iloc[st.session_state.current_idx]

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

    st.markdown(f"### Event #{st.session_state.current_idx:,} / {len(st.session_state.messages):,}")
    st.markdown(f"**Event Type:** :{msg_type_color}[{msg_type_text}]")

    with st.spinner("Reconstructing order book state..."):
        order_book = _ensure_order_book_state(
            st.session_state.messages, st.session_state.current_idx
        )

    st.markdown("---")
    st.markdown("### Current Order Book State")

    total_bid_volume = sum(o["size"] for o in order_book["bids"].values())
    total_ask_volume = sum(o["size"] for o in order_book["asks"].values())

    render_metrics_grid(
        [
            ("Total Bid Orders", f"{len(order_book['bids']):,}"),
            ("Total Ask Orders", f"{len(order_book['asks']):,}"),
            ("Bid Volume", f"{total_bid_volume:,}"),
            ("Ask Volume", f"{total_ask_volume:,}"),
        ],
        columns=2,
    )

    st.markdown("---")

    st.markdown("### Order Book Depth with Queue Details")
    fig_depth, bid_levels, ask_levels = plot_order_book_depth_with_queue(
        order_book, levels=display_levels
    )
    st.plotly_chart(fig_depth, use_container_width=True, key=f"l3_depth_{st.session_state.current_idx}")

    st.markdown("#### Order Queue Details")
    date_str = get_dataset_date(st.session_state.selected_ticker)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Bid Queue (FIFO Order)**")
        bid_table = format_order_queue_table(bid_levels, "bid", date_str)
        if not bid_table.empty:
            st.dataframe(bid_table, use_container_width=True, hide_index=True)
        else:
            st.write("No bid orders")

    with col2:
        st.markdown("**Ask Queue (FIFO Order)**")
        ask_table = format_order_queue_table(ask_levels, "ask", date_str)
        if not ask_table.empty:
            st.dataframe(ask_table, use_container_width=True, hide_index=True)
        else:
            st.write("No ask orders")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Order Size Distribution")
        fig_sizes = plot_order_size_distribution(order_book)
        st.plotly_chart(fig_sizes, use_container_width=True, config={})

    with col2:
        st.markdown("### Order Arrival Rate")
        fig_flow = plot_order_flow_rate(st.session_state.messages, window=100, date_str=date_str)
        st.plotly_chart(fig_flow, use_container_width=True, config={})

    st.markdown("---")

    st.markdown("### Individual Order Tracker")
    order_id_input = st.number_input(
        "Enter Order ID to track", min_value=0, value=0, step=1, key="l3_order_id"
    )

    if order_id_input > 0:
        fig_timeline = plot_order_timeline(st.session_state.messages, order_id_input, date_str)
        st.plotly_chart(fig_timeline, use_container_width=True, config={})

    with st.expander("View Complete Order Book (L3)"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**All Bid Orders**")
            if order_book["bids"]:
                bid_df = pd.DataFrame.from_dict(order_book["bids"], orient="index")
                bid_df = bid_df.sort_values("price", ascending=False)
                st.dataframe(bid_df, use_container_width=True)
            else:
                st.write("No bid orders")

        with col2:
            st.markdown("**All Ask Orders**")
            if order_book["asks"]:
                ask_df = pd.DataFrame.from_dict(order_book["asks"], orient="index")
                ask_df = ask_df.sort_values("price")
                st.dataframe(ask_df, use_container_width=True)
            else:
                st.write("No ask orders")


def show():
    apply_global_styles()
    st.title("L3")
    st.markdown(
        "**Market by order: Order Book Visualizer with order queue position tracking**"
    )

    init_session_state()

    if "order_book" not in st.session_state:
        st.session_state.order_book = None
    if "order_book_last_idx" not in st.session_state:
        st.session_state.order_book_last_idx = None

    messages, orderbook, available_tickers = load_ticker_data()

    if messages is None:
        st.error(
            "No LOBSTER data files found. Please ensure sample data directories exist."
        )
        return

    display_levels = render_l3_sidebar()
    render_main_content(st.session_state, display_levels)
