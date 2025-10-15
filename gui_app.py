import streamlit as st
from gui_pages import l1_page, l2_page, l3_page

st.set_page_config(
    page_title="LOBSTER Order Book Visualizer",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main():
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

    st.sidebar.title("LOBSTER Visualizer")

    page = st.sidebar.radio(
        "Select View",
        ["L1 - Price Action", "L2 - Price Levels", "L3 - Order Flow"],
        index=0,
    )

    st.sidebar.markdown("---")

    if page == "L1 - Price Action":
        l1_page.show()
    elif page == "L2 - Price Levels":
        l2_page.show()
    elif page == "L3 - Order Flow":
        l3_page.show()


if __name__ == "__main__":
    main()
