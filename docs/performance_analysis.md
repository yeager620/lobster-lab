# Performance Analysis and Optimization Plan

## Summary of Observed Bottlenecks

### Data acquisition and preparation
- `load_data_from_hf` and `load_data` download or read the raw CSV pairs with `read_lobster(..., as_pandas=False)` but then eagerly convert the Polars `DataFrame` objects to pandas via `.to_pandas()` before caching in Streamlit, forcing a full materialization of every column and losing Polars' lazy/parallel execution advantages.【F:gui_pages/data.py†L47-L69】
- Each rerun of the Streamlit script invalidates large objects stored in `st.session_state` (messages/orderbook) when navigation widgets call `st.rerun()`, which triggers the download/parse pipeline again if `data_loaded` is reset, causing repeated IO and conversion costs.【F:gui_pages/data.py†L103-L146】【F:gui_pages/ui.py†L65-L132】
- Hugging Face downloads happen file-by-file with `hf_hub_download`, so switching tickers repeatedly causes redundant downloads instead of using a repository snapshot or streaming interface that can reuse cached artifacts efficiently.【F:gui_pages/data.py†L47-L63】

### Transformation logic
- L1 visualizations recompute aggregations like `create_candlestick_data` and `compute_volume_profile` on the full window every rerun using pandas groupby operations; these scans scale linearly with the window size and happen on the Python interpreter, leaving Polars' SIMD and multi-threading unused.【F:gui_pages/l1_page.py†L18-L120】【F:gui_pages/l1_page.py†L122-L214】
- L2 plotting builds bid/ask depth DataFrames row-by-row in Python loops and re-creates Plotly figures for every navigation step, which is CPU-bound when traversing tens of levels.【F:gui_pages/l2_page.py†L18-L112】【F:gui_pages/l2_page.py†L148-L188】
- L3 reconstruction walks through up to 50k messages on each cache miss in `reconstruct_order_book_state`, and the cache key is the entire `messages` DataFrame, so any change to the Streamlit cache inputs causes the O(n) replay to execute again.【F:gui_pages/l3_page.py†L39-L105】【F:gui_pages/l3_page.py†L532-L599】

### UI rendering and interactivity
- Sidebar widgets trigger `st.rerun()` frequently, invalidating charts and cached computations instead of updating state variables directly; Plotly figures are regenerated even when underlying slices are unchanged.【F:gui_pages/ui.py†L65-L145】【F:gui_pages/l1_page.py†L150-L214】【F:gui_pages/l2_page.py†L132-L188】
- Large DataFrames are passed to `st.dataframe` without pagination or downcasting, forcing Streamlit to serialize and transfer full tables to the browser on every rerun.【F:gui_pages/l1_page.py†L196-L214】【F:gui_pages/l2_page.py†L188-L199】【F:gui_pages/l3_page.py†L599-L683】

## Recommended Optimizations

### Leverage Polars end-to-end
- Keep message/orderbook data as Polars objects in session state and expose pandas views only for components that strictly need them, minimizing memory copies and unlocking Polars' vectorized arithmetic and eager/lazy APIs.【F:gui_pages/data.py†L47-L86】
- Rewrite aggregation helpers (`create_candlestick_data`, `compute_volume_profile`, `get_orderbook_depth`) to operate on Polars expressions or lazy plans so grouping, filtering, and rolling computations execute in parallel; cache compiled lazy plans and only collect the slices needed for the active window.【F:gui_pages/l1_page.py†L18-L214】【F:gui_pages/l2_page.py†L18-L112】
- Downcast numeric columns (e.g., prices to `Int32`, sizes to `Int32/UInt32`) when loading data to reduce transfer size before converting to pandas for Plotly, or supply Polars columns directly to Plotly by converting only the required series.

### Streamlit performance patterns
- Replace frequent `st.rerun()` calls with state mutations that update widgets (`st.session_state` callbacks) and isolate expensive computations inside `st.cache_resource` or `st.cache_data` functions keyed by immutable parameters instead of entire DataFrames.【F:gui_pages/ui.py†L65-L145】【F:gui_pages/l3_page.py†L532-L599】
- Use `st.experimental_fragment` or memoized helper functions for charts so only the section that depends on the current index re-renders, and reuse Plotly figures by updating traces when possible to avoid full JSON serialization.
- Paginate large tables with `st.dataframe(..., height=...)` and provide Polars `.limit()` slices to keep the browser payload small, or summarize levels in Streamlit metrics while exposing full data via download buttons.

### Hugging Face Hub utilization
- Switch from per-file `hf_hub_download` calls to `snapshot_download` to fetch the entire repo once and rely on the local cache thereafter; expose dataset revision pins to avoid repeated 404 checks and to leverage file-system caching across reruns.【F:gui_pages/data.py†L47-L69】
- For large artifacts, adopt the Hub's streaming (`hf_hub_download` with `local_files_only=True` after initial sync) or dataset streaming APIs so the GUI only loads the requested window instead of the full trading day.
- Parallelize initial downloads by prefetching both message and order-book files asynchronously (e.g., using `asyncio` or a thread pool) and show progress bars fed by the hub's callback hooks for better UX during large transfers.

### Order-book specific strategies
- Incrementally maintain the reconstructed L3 order book in Polars or a dedicated state object instead of reprocessing 50k rows on cache invalidation; persist computed book states for key indices to allow binary search/time jumping without replaying from the beginning.【F:gui_pages/l3_page.py†L39-L105】【F:gui_pages/l3_page.py†L532-L599】
- Precompute snapshots (e.g., every 1,000 events) offline and store them as Parquet/IPC in the dataset so the GUI can load a near-by baseline and replay a small delta range when users jump around.
- Use vectorized numpy/polars operations to build depth charts rather than Python loops per level, enabling faster updates and clearer separation between data preparation and visualization logic.【F:gui_pages/l2_page.py†L18-L112】【F:gui_pages/l3_page.py†L150-L265】

## Technology Notes

- **Polars**: Offers a Rust-backed query engine with lazy evaluation, automatic query optimization, and multi-threaded execution, which can yield 10× speedups over pandas on wide aggregations and join-heavy workloads when data is kept in Polars throughout the pipeline.
- **Streamlit**: Its caching decorators memoize pure functions; combining `st.cache_resource` for connectors and `st.cache_data` for immutable data slices reduces redundant reruns. Fragmented rendering (`st.experimental_fragment`) can keep UI responsive even when heavy computations occur elsewhere.
- **Hugging Face Hub**: Dataset repos automatically cache downloads under `~/.cache/huggingface`. Using `snapshot_download` or dataset streaming prevents repeated network transfers and enables version pinning via revisions, improving reproducibility in dashboards.

## Implemented optimizations

- **Polars-first analytics**: Data is now held as Polars DataFrames inside session state and reused for heavy aggregations. L1 candlestick and volume profile builders rely on Polars group-bys and an LRU memoization layer, so repeated navigation reuses cached slices instead of recomputing pandas group-bys from scratch.【F:gui_pages/data.py†L1-L210】【F:gui_pages/l1_page.py†L1-L231】
- **Efficient data access**: Hugging Face downloads use `snapshot_download` once per ticker and reuse a shared cache, while local file loads retain optimized Polars frames and lazily materialize pandas views only when required by the UI.【F:gui_pages/data.py†L36-L143】
- **Order book replay cache**: L3 reconstruction keeps an incremental replay cache bound by a configurable lookback, eliminating the O(n) full replay on every interaction and dramatically reducing CPU time when scrubbing through events.【F:gui_pages/l3_page.py†L51-L111】
- **Depth snapshot caching**: L2 depth extraction reuses the underlying Polars slice per index and memoizes results per ticker/level combination to minimize redundant Series traversals when stepping through events.【F:gui_pages/l2_page.py†L1-L111】

