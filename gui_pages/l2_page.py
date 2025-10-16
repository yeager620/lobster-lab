import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from typing import Optional, Tuple
from .shared import (
    init_session_state,
    load_ticker_data,
    seconds_to_eastern_time,
    get_dataset_date,
    apply_global_styles,
    render_metrics_grid,
)


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
                y=bids["price"].astype(float),
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
                y=asks["price"].astype(float),
                orientation="h",
                name="Asks",
                marker=dict(color="rgba(255, 50, 50, 0.7)"),
                hovertemplate="<b>Ask</b><br>Price: $%{y:.4f}<br>Size: %{x:,}<extra></extra>",
            )
        )

    if current_msg is not None and current_msg["type"] in [4, 5]:
        exec_price = float(current_msg["price"]) / 10000.0
        exec_size = int(current_msg["size"]) if pd.notna(current_msg["size"]) else 0
        direction_val = (
            int(current_msg["direction"]) if pd.notna(current_msg["direction"]) else 0
        )
        direction = "BUY" if direction_val == -1 else "SELL"
        fig.add_hline(
            y=float(exec_price),
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


def format_message_details(msg: pd.Series, date_str: str = "2012-06-21") -> dict:
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
        "Time": seconds_to_eastern_time(msg["time"], date_str),
        "Type": message_types.get(int(msg["type"]), "Unknown"),
        "Order ID": f"{int(msg['order_id']):,}",
        "Size": f"{int(msg['size']):,}",
        "Price": f"${msg['price'] / 10000.0:.2f}",
        "Side": direction_map.get(int(msg["direction"]), "?"),
    }


def show():
    apply_global_styles()
    st.title("L2")
    st.markdown("**L2 Order Book Visualizer: Aggregated liquidity by price**")

    init_session_state()

    messages, orderbook, available_tickers = load_ticker_data()

    if messages is None:
        st.error(
            "No LOBSTER data files found. Please ensure sample data directories exist."
        )
        return

    n_levels = len([col for col in orderbook.columns if col.startswith("ask_price_")])

    st.sidebar.header("Playback Controls")

    max_idx = len(messages) - 1
    if st.session_state.current_idx > max_idx:
        st.session_state.current_idx = max_idx
    if st.session_state.current_idx < 0:
        st.session_state.current_idx = 0

    step_size = st.sidebar.selectbox(
        "Step Size", [1, 10, 100, 1000], index=0, key="step_size_selector"
    )

    st.sidebar.markdown("**Navigation**")
    nav_col1, nav_col2 = st.sidebar.columns([1, 1])

    with nav_col1:
        if st.button(f"-{step_size}", key="btn_back", use_container_width=True):
            st.session_state.current_idx = max(
                0, st.session_state.current_idx - step_size
            )
            st.rerun()

        if st.button("-1", key="btn_back_one", use_container_width=True):
            st.session_state.current_idx = max(0, st.session_state.current_idx - 1)
            st.rerun()

    with nav_col2:
        if st.button(f"+{step_size}", key="btn_next", use_container_width=True):
            st.session_state.current_idx = min(
                max_idx, st.session_state.current_idx + step_size
            )
            st.rerun()

        if st.button("+1", key="btn_next_one", use_container_width=True):
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
        if st.button("Prev", key="btn_prev_event", use_container_width=True):
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
        if st.button("Next", key="btn_next_event", use_container_width=True):
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

    if n_levels > 1:
        display_levels = st.sidebar.slider(
            "Order Book Levels to Display",
            min_value=1,
            max_value=n_levels,
            value=min(10, n_levels),
            step=1,
            key="levels_slider",
        )
    else:
        st.sidebar.info(f"Dataset has {n_levels} level only")
        display_levels = 1

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
    date_str = get_dataset_date(st.session_state.selected_ticker)
    msg_details = format_message_details(current_msg, date_str)

    render_metrics_grid(list(msg_details.items()), columns=3)

    st.markdown("---")

    st.markdown("#### Order Book Statistics")
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

    st.markdown("#### Order Book Depth")
    bids, asks = get_orderbook_depth(current_ob, levels=display_levels)

    if not bids.empty and not asks.empty:
        fig_ob = plot_orderbook(bids, asks, current_msg)
        st.plotly_chart(
            fig_ob, use_container_width=True, key=f"orderbook_chart_{current_idx}"
        )
    else:
        st.warning("No valid order book data at this index.")

    with st.expander("View Raw Data"):
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Message Data**")
            st.dataframe(current_msg.to_frame(), use_container_width=True)
        with col2:
            st.write("**Order Book Data (First 10 levels)**")
            ob_display = current_ob.to_frame().head(40)
            st.dataframe(ob_display, use_container_width=True)
