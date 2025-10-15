"""
LOBSTER CSV parser with Polars

Implements high-performance parsers for LOBSTER message and orderbook CSV files according to
"LOBSTER_SampleFiles_ReadMe.txt" contained in the sample folders.

Spec summary (from the ReadMe):
- Message file (Nx6):
    1) Time (seconds after midnight, decimal; ms to ns precision)
    2) Type (1 new, 2 partial cancel, 3 delete, 4 exec visible, 5 exec hidden, 7 halt indicator)
    3) Order ID (unique reference)
    4) Size (shares)
    5) Price (dollars * 10000; e.g., $91.14 -> 911400)
    6) Direction (-1 sell, 1 buy)

- Orderbook file (Nx(4*L)) with L levels (occupied price levels):
    For level i starting at 1: [ask_price_i, ask_size_i, bid_price_i, bid_size_i]
    Unoccupied price levels are filled with dummy values: ask price 9999999999, bid price -9999999999, sizes 0.

This module exposes:
- parse_message_file(path, time_as='float'): read full message file to Polars DataFrame.
- parse_orderbook_file(path, levels=None): read full orderbook file to Polars DataFrame.
- iter_lobster(message_path, orderbook_path, chunksize=100_000, time_as='float'):
    iterate the two files in lockstep yielding chunked DataFrames with aligned rows.
- read_lobster(message_path, orderbook_path, time_as='float', as_pandas=False):
    convenience function to read entire files.

Notes:
- By default, returns Polars DataFrames for performance.
- Set as_pandas=True to get pandas DataFrames (for visualization compatibility).
- The timestamp is parsed as float seconds since midnight by default.
"""

from __future__ import annotations

from typing import Generator, List, Optional, Tuple, Any
import polars as pl


# ---------------------------
# Helpers
# ---------------------------


def _infer_levels_from_orderbook_columns(n_cols: int) -> int:
    if n_cols % 4 != 0:
        raise ValueError(
            f"Orderbook file must have columns multiple of 4, got {n_cols}."
        )
    return n_cols // 4


def orderbook_column_names(levels: int) -> List[str]:
    names: List[str] = []
    for i in range(1, levels + 1):
        names.extend(
            [f"ask_price_{i}", f"ask_size_{i}", f"bid_price_{i}", f"bid_size_{i}"]
        )
    return names


MESSAGE_COLUMNS = [
    "time",
    "type",
    "order_id",
    "size",
    "price",
    "direction",
]

MESSAGE_SCHEMA = {
    "time": pl.Float64,
    "type": pl.Int8,
    "order_id": pl.Int64,
    "size": pl.Int32,
    "price": pl.Int64,
    "direction": pl.Int8,
}


# ---------------------------
# Public API
# ---------------------------


def parse_message_file(path: str, time_as: str = "float") -> pl.DataFrame:
    """
    Parse a LOBSTER message CSV into a Polars DataFrame.

    Parameters:
    - path: path to the ..._message_LEVEL.csv file
    - time_as: 'float' (default) parses time to float64 seconds; 'str' preserves raw string.

    Returns: Polars DataFrame with columns [time, type, order_id, size, price, direction]
    """
    if time_as not in ("float", "str"):
        raise ValueError("time_as must be 'float' or 'str'")

    schema = MESSAGE_SCHEMA.copy()
    if time_as == "str":
        schema["time"] = pl.Utf8

    df = pl.read_csv(
        path,
        has_header=False,
        new_columns=MESSAGE_COLUMNS,
        schema=schema,
    )
    return df


def parse_orderbook_file(path: str, levels: Optional[int] = None) -> pl.DataFrame:
    """
    Parse a LOBSTER orderbook CSV into a Polars DataFrame.

    Parameters:
    - path: path to the ..._orderbook_LEVEL.csv file
    - levels: number of levels; if None, inferred from number of columns (must be multiple of 4)

    Returns: Polars DataFrame with columns [ask_price_1, ask_size_1, bid_price_1, bid_size_1, ..., ..._L]
    """
    # Read one row to infer number of columns
    preview = pl.read_csv(path, has_header=False, n_rows=1)
    n_cols = len(preview.columns)

    inferred_levels = _infer_levels_from_orderbook_columns(n_cols)
    L = levels or inferred_levels
    if L != inferred_levels:
        raise ValueError(
            f"Provided levels={levels} does not match file columns={n_cols} (implies {inferred_levels} levels)."
        )

    names = orderbook_column_names(L)

    # Read full file with proper names
    df = pl.read_csv(
        path,
        has_header=False,
        new_columns=names,
    )
    return df


def iter_lobster(
    message_path: str,
    orderbook_path: str,
    *,
    chunksize: int = 100_000,
    time_as: str = "float",
) -> Generator[Tuple[pl.DataFrame, pl.DataFrame], None, None]:
    """
    Iterate over message and orderbook files in lockstep, yielding aligned chunks.

    This is memory-efficient for large days. Each yielded tuple is (message_chunk, orderbook_chunk),
    where both DataFrames have the same number of rows corresponding to the same event indices.

    Parameters:
    - message_path: path to message CSV
    - orderbook_path: path to orderbook CSV
    - chunksize: number of rows per chunk to yield
    - time_as: 'float' or 'str' (see parse_message_file)
    """
    if time_as not in ("float", "str"):
        raise ValueError("time_as must be 'float' or 'str'")

    # Prepare schema for messages
    schema = MESSAGE_SCHEMA.copy()
    if time_as == "str":
        schema["time"] = pl.Utf8

    # Infer orderbook columns
    ob_preview = pl.read_csv(orderbook_path, has_header=False, n_rows=1)
    L = _infer_levels_from_orderbook_columns(len(ob_preview.columns))
    ob_names = orderbook_column_names(L)

    # Read files with batched reader
    msg_batches = pl.read_csv_batched(
        message_path,
        has_header=False,
        new_columns=MESSAGE_COLUMNS,
        schema=schema,
        batch_size=chunksize,
    )

    ob_batches = pl.read_csv_batched(
        orderbook_path,
        has_header=False,
        new_columns=ob_names,
        batch_size=chunksize,
    )

    while True:
        msg_chunk = msg_batches.next_batches(1)
        ob_chunk = ob_batches.next_batches(1)

        if not msg_chunk or not ob_chunk:
            break

        msg_df = msg_chunk[0]
        ob_df = ob_chunk[0]

        # Sanity check: ensure same number of rows
        if len(msg_df) != len(ob_df):
            raise ValueError(
                f"Chunk misalignment: message rows {len(msg_df)} != orderbook rows {len(ob_df)}.\n"
                f"Ensure files correspond to the same ticker/day/level."
            )

        yield msg_df, ob_df


def read_lobster(
    message_path: str,
    orderbook_path: str,
    *,
    time_as: str = "float",
    as_pandas: bool = False,
) -> Tuple[Any, Any]:
    """
    Convenience function to read the entire day into memory, returning (messages, orderbook).

    Parameters:
    - message_path: path to message CSV
    - orderbook_path: path to orderbook CSV
    - time_as: 'float' or 'str'
    - as_pandas: if True, convert to pandas DataFrames (for visualization); default False returns Polars

    Returns: Tuple of (messages, orderbook) as either Polars or pandas DataFrames

    For large files prefer iter_lobster(...).
    """
    messages = parse_message_file(message_path, time_as=time_as)
    orderbook = parse_orderbook_file(orderbook_path)

    if len(messages) != len(orderbook):
        raise ValueError(
            f"Row count mismatch: message rows {len(messages)} != orderbook rows {len(orderbook)}.\n"
            f"Ensure files correspond to the same ticker/day/level."
        )

    if as_pandas:
        return messages.to_pandas(), orderbook.to_pandas()

    return messages, orderbook


if __name__ == "__main__":
    # Minimal usage example on the included sample files.
    # Adjust paths/ticker as needed.
    aapl_dir = "LOBSTER_SampleFile_AAPL_2012-06-21_50"
    msg_path = f"{aapl_dir}/AAPL_2012-06-21_34200000_37800000_message_50.csv"
    ob_path = f"{aapl_dir}/AAPL_2012-06-21_34200000_37800000_orderbook_50.csv"
    try:
        msgs, obs = read_lobster(msg_path, ob_path)
        print("Loaded:", len(msgs), "events,", len(obs.columns) // 4, "levels")
        print(msgs.head())
        print(obs.head())
    except FileNotFoundError:
        # If run from a different CWD, suppress sample demo
        pass
