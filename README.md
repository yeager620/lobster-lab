# LOBSTER Order Book Visualizer

Interactive visualization and backtesting toolkit for LOBSTER limit order book data with dedicated L1, L2, and L3 market microstructure views.

[**Launch Streamlit App**](https://lobster-lab.streamlit.app/)

[**Hugging Face Spaces**](https://huggingface.co/spaces/totalorganfailure/lobster-lab)

## Highlights
- Load sample LOBSTER datasets directly from Hugging Face or local CSV exports.
- Navigate events with customizable integer step sizes and jump controls shared across pages.
- Animate L2 and L3 views with a play/pause toggle that advances through the order flow automatically.
- Inspect best bid/ask evolution, depth imbalances, and queue level details in real time.

## Local Development
1. Install dependencies with `pip install -r requirements.txt` (or `pip install -e .`).
2. Launch the Streamlit experience via `streamlit run gui_app.py`.
3. Optionally export the `HF_REPO_ID` environment variable to pull datasets from Hugging Face.

