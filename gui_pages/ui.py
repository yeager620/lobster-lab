import time
import streamlit as st
from typing import Tuple, Any, Iterable


def apply_global_styles() -> None:
    global_style = """
    <style>
    :root {
        --lobster-lab-h1: clamp(2.0rem, 2.4vw + 0.8rem, 3.2rem);
        --lobster-lab-h2: clamp(1.6rem, 1.9vw + 0.6rem, 2.4rem);
        --lobster-lab-h3: clamp(1.35rem, 1.4vw + 0.55rem, 1.9rem);
        --lobster-lab-text: clamp(0.95rem, 0.95vw + 0.55rem, 1.1rem);
    }

    h1, h2, h3, h4, h5, h6 {
        font-weight: 600 !important;
        line-height: 1.2 !important;
        margin-bottom: 0.75rem !important;
    }

    h1 { font-size: var(--lobster-lab-h1) !important; }
    h2 { font-size: var(--lobster-lab-h2) !important; }
    h3 { font-size: var(--lobster-lab-h3) !important; }
    p, li, span, label {
        font-size: var(--lobster-lab-text) !important;
        line-height: 1.5 !important;
    }

    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label {
        white-space: normal !important;
    }

    [data-testid="stMetricValue"] {
        font-size: clamp(1.1rem, 1.4vw + 0.6rem, 1.8rem) !important;
        line-height: 1.15 !important;
        word-break: break-word !important;
    }

    [data-testid="stMetricLabel"] {
        font-size: clamp(0.75rem, 0.8vw + 0.45rem, 1.05rem) !important;
        white-space: normal !important;
        line-height: 1.2 !important;
    }

    section[data-testid="stHorizontalBlock"] > div {
        padding-top: 0.25rem !important;
        padding-bottom: 0.25rem !important;
    }
    </style>
    """

    st.markdown(global_style, unsafe_allow_html=True)

def render_metrics_grid(
    metrics: Iterable[Tuple[str, Any]],
    *,
    columns: int = 3,
) -> None:
    items = list(metrics)
    if not items:
        return

    columns = max(1, columns)

    for start in range(0, len(items), columns):
        row_items = items[start : start + columns]
        row_columns = st.columns(len(row_items))
        for col, (label, value) in zip(row_columns, row_items):
            with col:
                st.metric(label, value)

def render_sidebar(*, playback_enabled: bool = True) -> None:
    st.sidebar.header("Playback Controls")

    max_idx = len(st.session_state.messages) - 1
    if st.session_state.current_idx > max_idx:
        st.session_state.current_idx = max_idx
    if st.session_state.current_idx < 0:
        st.session_state.current_idx = 0

    max_step = max_idx + 1 if max_idx >= 0 else 1
    existing_step = int(st.session_state.get("step_size_selector", 1) or 1)
    bounded_step = min(max(existing_step, 1), max_step)
    st.session_state.step_size_selector = bounded_step
    step_size = int(
        st.sidebar.number_input(
            "Step Size",
            min_value=1,
            max_value=max_step,
            value=bounded_step,
            step=1,
            key="step_size_selector",
        )
    )

    if playback_enabled:
        if "is_playing" not in st.session_state:
            st.session_state.is_playing = False
        if "playback_interval" not in st.session_state:
            st.session_state.playback_interval = 0.5

        play_label = "Pause" if st.session_state.is_playing else "Play"
        if st.sidebar.button(
            play_label, key="play_pause_button", use_container_width=True
        ):
            st.session_state.is_playing = not st.session_state.is_playing
            st.rerun()

    st.sidebar.markdown("**Navigation**")
    nav_col1, nav_col2 = st.sidebar.columns([1, 1])

    with nav_col1:
        if st.button(f"-{step_size}", key="btn_back", use_container_width=True):
            st.session_state.current_idx = max(
                0, st.session_state.current_idx - step_size
            )
            st.session_state.is_playing = False
            st.rerun()

        if st.button("-1", key="btn_back_one", use_container_width=True):
            st.session_state.current_idx = max(0, st.session_state.current_idx - 1)
            st.session_state.is_playing = False
            st.rerun()

    with nav_col2:
        if st.button(f"+{step_size}", key="btn_next", use_container_width=True):
            st.session_state.current_idx = min(
                max_idx, st.session_state.current_idx + step_size
            )
            st.session_state.is_playing = False
            st.rerun()

        if st.button("+1", key="btn_next_one", use_container_width=True):
            st.session_state.current_idx = min(
                max_idx, st.session_state.current_idx + 1
            )
            st.session_state.is_playing = False
            st.rerun()

    if st.sidebar.button("Reset to Start", key="btn_reset", use_container_width=True):
        st.session_state.current_idx = 0
        st.session_state.is_playing = False
        st.rerun()

    if playback_enabled and st.session_state.is_playing and max_idx >= 0:
        interval = max(float(st.session_state.playback_interval), 0.1)
        time.sleep(interval)
        next_idx = min(max_idx, st.session_state.current_idx + step_size)
        if next_idx == st.session_state.current_idx:
            st.session_state.is_playing = False
        else:
            st.session_state.current_idx = next_idx
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
                if int(st.session_state.messages.iloc[i]["type"]) == target_type:
                    found_idx = i
                    break

            if found_idx is not None:
                st.session_state.current_idx = found_idx
                st.session_state.last_search_result = (
                    f"Jumped to {event_type[1]} at #{found_idx}"
                )
                st.session_state.is_playing = False
                st.rerun()
            else:
                st.session_state.last_search_result = (
                    f"No previous {event_type[1]} found"
                )

    with event_col2:
        if st.button("Next", key="btn_next_event", use_container_width=True):
            target_type = event_type[0]
            found_idx = None
            for i in range(st.session_state.current_idx + 1, len(st.session_state.messages)):
                if int(st.session_state.messages.iloc[i]["type"]) == target_type:
                    found_idx = i
                    break

            if found_idx is not None:
                st.session_state.current_idx = found_idx
                st.session_state.last_search_result = (
                    f"Jumped to {event_type[1]} at #{found_idx}"
                )
                st.session_state.is_playing = False
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
            st.session_state.is_playing = False
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
        st.session_state.is_playing = False
        st.rerun()
