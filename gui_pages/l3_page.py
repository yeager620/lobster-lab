import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.colors as pc
from typing import Dict, List, Tuple
from collections import defaultdict
from .shared import init_session_state, load_ticker_data


def reconstruct_order_book_state(messages: pd.DataFrame, up_to_idx: int) -> Dict:
    order_book = {"bids": {}, "asks": {}}

    for i in range(min(up_to_idx + 1, len(messages))):
        msg = messages.iloc[i]
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
        for price, orders in bid_levels:
            orders_sorted = sorted(orders, key=lambda x: x["time"])
            cumulative_size = 0
            queue_length = len(orders_sorted)
            for queue_idx, order in enumerate(orders_sorted):
                if queue_length > 1:
                    scale_value = queue_idx / (queue_length - 1)
                else:
                    scale_value = 0
                color = pc.sample_colorscale("Greens", scale_value)[0]
                hover_text = (
                    f"<b>Bid @ ${price:.2f}</b><br>"
                    f"Queue Pos: {queue_idx + 1}/{queue_length}<br>"
                    f"Order ID: {order['id']}<br>"
                    f"Size: {order['size']:,}<br>"
                    f"Time: {order['time']:.2f}s"
                )

                fig.add_trace(
                    go.Bar(
                        x=[order["size"]],
                        y=[price],
                        base=[cumulative_size],
                        orientation="h",
                        name="Bids" if queue_idx == 0 else None,
                        marker=dict(color=color),
                        hovertemplate="%{customdata}<extra></extra>",
                        customdata=[hover_text],
                        legendgroup="bids",
                        showlegend=queue_idx == 0,
                    )
                )
                cumulative_size += order["size"]

    if ask_levels:
        for price, orders in ask_levels:
            orders_sorted = sorted(orders, key=lambda x: x["time"])
            cumulative_size = 0
            queue_length = len(orders_sorted)
            for queue_idx, order in enumerate(orders_sorted):
                if queue_length > 1:
                    scale_value = queue_idx / (queue_length - 1)
                else:
                    scale_value = 0
                color = pc.sample_colorscale("Reds", scale_value)[0]
                hover_text = (
                    f"<b>Ask @ ${price:.2f}</b><br>"
                    f"Queue Pos: {queue_idx + 1}/{queue_length}<br>"
                    f"Order ID: {order['id']}<br>"
                    f"Size: {order['size']:,}<br>"
                    f"Time: {order['time']:.2f}s"
                )

                fig.add_trace(
                    go.Bar(
                        x=[order["size"]],
                        y=[price],
                        base=[cumulative_size],
                        orientation="h",
                        name="Asks" if queue_idx == 0 else None,
                        marker=dict(color=color),
                        hovertemplate="%{customdata}<extra></extra>",
                        customdata=[hover_text],
                        legendgroup="asks",
                        showlegend=queue_idx == 0,
                    )
                )
                cumulative_size += order["size"]

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
        title="Order Book Depth by Queue Position",
        xaxis_title="Shares",
        yaxis_title="Price ($)",
        barmode="overlay",
        height=600,
        hovermode="closest",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=80, r=80, t=80, b=60),
        xaxis=dict(range=[0, max_size * 1.1] if max_size > 0 else [0, 100]),
    )

    return fig, bid_levels, ask_levels


def format_order_queue_table(price_levels: List, side: str) -> pd.DataFrame:
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
                    "Time": f"{order['time']:.2f}s",
                }
            )

    return pd.DataFrame(rows) if rows else pd.DataFrame()


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


def plot_order_timeline(messages: pd.DataFrame, order_id: int) -> go.Figure:
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
    sizes = order_msgs["size"].values
    prices = order_msgs["price"].values / 10000.0

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=times,
            y=sizes,
            mode="lines+markers",
            marker=dict(size=12, color="blue"),
            line=dict(color="blue", width=2),
            text=[
                f"Event: {e}<br>Time: {t:.2f}s<br>Size: {s}<br>Price: ${p:.2f}"
                for e, t, s, p in zip(events, times, sizes, prices)
            ],
            hovertemplate="%{text}<extra></extra>",
            name=f"Order {order_id}",
        )
    )

    fig.update_layout(
        title=f"Order Lifecycle: ID {order_id}",
        xaxis_title="Time (seconds after midnight)",
        yaxis_title="Remaining Size",
        height=400,
    )

    return fig


def plot_order_flow_rate(messages: pd.DataFrame, window: int = 100) -> go.Figure:
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
    rates = []

    for i in range(window, len(new_orders)):
        time_window = new_orders.iloc[i - window : i]
        time_diff = time_window["time"].iloc[-1] - time_window["time"].iloc[0]
        if time_diff > 0:
            rate = window / time_diff
            times.append(time_window["time"].iloc[-1])
            rates.append(rate)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=times,
            y=rates,
            mode="lines",
            line=dict(color="purple", width=2),
            name="Order Arrival Rate",
        )
    )

    fig.update_layout(
        title=f"Order Arrival Rate (Rolling Window: {window} orders)",
        xaxis_title="Time (seconds after midnight)",
        yaxis_title="Orders per Second",
        height=400,
    )

    return fig


def show():
    st.title("L3")
    st.markdown(
        "**Market by order: Order Book Visualizer with order queue position tracking**"
    )

    init_session_state()
    messages, orderbook, available_tickers = load_ticker_data()

    if messages is None:
        st.error(
            "No LOBSTER data files found. Please ensure sample data directories exist."
        )
        return

    st.sidebar.header("Playback Controls")
    max_idx = len(messages) - 1

    if st.session_state.current_idx > max_idx:
        st.session_state.current_idx = max_idx
    if st.session_state.current_idx < 0:
        st.session_state.current_idx = 0

    step_size = st.sidebar.selectbox(
        "Step Size", [1, 10, 100, 1000], index=0, key="l3_step_size_selector"
    )

    st.sidebar.markdown("**Navigation**")
    nav_col1, nav_col2 = st.sidebar.columns([1, 1])

    with nav_col1:
        if st.button(f"-{step_size}", key="l3_btn_back", use_container_width=True):
            st.session_state.current_idx = max(
                0, st.session_state.current_idx - step_size
            )
            st.rerun()

        if st.button("-1", key="l3_btn_back_one", use_container_width=True):
            st.session_state.current_idx = max(0, st.session_state.current_idx - 1)
            st.rerun()

    with nav_col2:
        if st.button(f"+{step_size}", key="l3_btn_next", use_container_width=True):
            st.session_state.current_idx = min(
                max_idx, st.session_state.current_idx + step_size
            )
            st.rerun()

        if st.button("+1", key="l3_btn_next_one", use_container_width=True):
            st.session_state.current_idx = min(
                max_idx, st.session_state.current_idx + 1
            )
            st.rerun()

    if st.sidebar.button(
        "Reset to Start", key="l3_btn_reset", use_container_width=True
    ):
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
        key="l3_event_type_selector",
    )

    event_col1, event_col2 = st.sidebar.columns(2)
    with event_col1:
        if st.button("Prev", key="l3_btn_prev_event", use_container_width=True):
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
        if st.button("Next", key="l3_btn_next_event", use_container_width=True):
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
        if st.sidebar.button("Clear message", key="l3_clear_search_msg"):
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
        key="l3_manual_idx_input",
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
        max_value=20,
        value=10,
        step=1,
        key="l3_levels_slider",
    )

    st.sidebar.markdown("---")

    current_idx = st.session_state.current_idx
    current_msg = messages.iloc[current_idx]

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

    with st.spinner("Reconstructing order book state..."):
        order_book = reconstruct_order_book_state(messages, current_idx)

    st.markdown("---")
    st.markdown("### Current Order Book State")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Bid Orders", len(order_book["bids"]))
    with col2:
        st.metric("Total Ask Orders", len(order_book["asks"]))
    with col3:
        total_bid_volume = sum(o["size"] for o in order_book["bids"].values())
        st.metric("Bid Volume", f"{total_bid_volume:,}")
    with col4:
        total_ask_volume = sum(o["size"] for o in order_book["asks"].values())
        st.metric("Ask Volume", f"{total_ask_volume:,}")

    st.markdown("---")

    st.markdown("### Order Book Depth with Queue Details")
    fig_depth, bid_levels, ask_levels = plot_order_book_depth_with_queue(
        order_book, levels=display_levels
    )
    st.plotly_chart(fig_depth, use_container_width=True, key=f"l3_depth_{current_idx}")

    st.markdown("#### Order Queue Details")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Bid Queue (FIFO Order)**")
        bid_table = format_order_queue_table(bid_levels, "bid")
        if not bid_table.empty:
            st.dataframe(bid_table, use_container_width=True, hide_index=True)
        else:
            st.write("No bid orders")

    with col2:
        st.markdown("**Ask Queue (FIFO Order)**")
        ask_table = format_order_queue_table(ask_levels, "ask")
        if not ask_table.empty:
            st.dataframe(ask_table, use_container_width=True, hide_index=True)
        else:
            st.write("No ask orders")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Order Size Distribution")
        fig_sizes = plot_order_size_distribution(order_book)
        st.plotly_chart(fig_sizes, use_container_width=True)

    with col2:
        st.markdown("### Order Arrival Rate")
        fig_flow = plot_order_flow_rate(messages, window=100)
        st.plotly_chart(fig_flow, use_container_width=True)

    st.markdown("---")

    st.markdown("### Individual Order Tracker")
    order_id_input = st.number_input(
        "Enter Order ID to track", min_value=0, value=0, step=1, key="l3_order_id"
    )

    if order_id_input > 0:
        fig_timeline = plot_order_timeline(messages, order_id_input)
        st.plotly_chart(fig_timeline, use_container_width=True)

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
