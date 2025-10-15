"""
Unified LOBSTER Order Book Visualizer

Multi-page Streamlit application for L2 (price-level) and L3 (order-level) visualization.

Run with: streamlit run gui_app.py
"""

import streamlit as st

# Configure page
st.set_page_config(
    page_title="LOBSTER Order Book Visualizer",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Import page modules
from gui_pages import l1_page, l2_page, l3_page


def main():
    # Custom CSS to make sidebar wider
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] {
            min-width: 350px;
            max-width: 350px;
        }
        </style>
    """,
        unsafe_allow_html=True,
    )

    # Sidebar navigation
    st.sidebar.title("LOBSTER Visualizer")

    page = st.sidebar.radio(
        "Select View",
        ["L1 - Price Action", "L2 - Price Levels", "L3 - Order Flow"],
        index=0,
    )

    st.sidebar.markdown("---")

    # Display selected page
    if page == "L1 - Price Action":
        l1_page.show()
    elif page == "L2 - Price Levels":
        l2_page.show()
    elif page == "L3 - Order Flow":
        l3_page.show()


if __name__ == "__main__":
    main()
