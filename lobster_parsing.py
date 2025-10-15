"""
LOBSTER CSV parser

Implements parsers for LOBSTER message and orderbook CSV files according to
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
- parse_message_file(path, time_as='float'): read full message file to pandas DataFrame.
- parse_orderbook_file(path, levels=None): read full orderbook file to pandas DataFrame.
- iter_lobster(message_path, orderbook_path, chunksize=100_000, time_as='float'):
    iterate the two files in lockstep yielding chunked DataFrames with aligned rows.

Notes:
- By default, the timestamp is parsed as float seconds since midnight. For maximum precision,
  you can pass time_as='str' to keep the raw string. (Floats may lose nanosecond precision.)
- Functions avoid adding headers automatically as the raw files have no header line.
"""

from __future__ import annotations

from typing import Generator, List, Optional, Tuple, Any

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

MESSAGE_DTYPES_FLOAT_TIME = {
    # Keep time as float for convenience by default
    "time": "float64",
    "type": "int8",
    "order_id": "int64",
    "size": "int32",
    "price": "int64",
    "direction": "int8",
}

MESSAGE_DTYPES_STR_TIME = {
    # Keep time as str to preserve full precision
    "time": "string",
    "type": "int8",
    "order_id": "int64",
    "size": "int32",
    "price": "int64",
    "direction": "int8",
}


# ---------------------------
# Public API
# ---------------------------


def parse_message_file(path: str, time_as: str = "float") -> Any:
    """
    Parse a LOBSTER message CSV into a pandas DataFrame.

    Parameters:
    - path: path to the ..._message_LEVEL.csv file
    - time_as: 'float' (default) parses time to float64 seconds; 'str' preserves raw string.

    Returns: DataFrame with columns [time, type, order_id, size, price, direction]
    """
    if time_as not in ("float", "str"):
        raise ValueError("time_as must be 'float' or 'str'")

    try:
        import pandas as pd  # Lazy import to avoid hard dependency at module import time
    except ImportError as e:
        raise ImportError(
            "parse_message_file requires pandas. Please install it: pip install pandas"
        ) from e

    dtypes = (
        MESSAGE_DTYPES_FLOAT_TIME if time_as == "float" else MESSAGE_DTYPES_STR_TIME
    )

    df = pd.read_csv(
        path,
        header=None,
        names=MESSAGE_COLUMNS,
        dtype=dtypes,
    )
    return df


def parse_orderbook_file(path: str, levels: Optional[int] = None) -> Any:
    """
    Parse a LOBSTER orderbook CSV into a pandas DataFrame.

    Parameters:
    - path: path to the ..._orderbook_LEVEL.csv file
    - levels: number of levels; if None, inferred from number of columns (must be multiple of 4)

    Returns: DataFrame with columns [ask_price_1, ask_size_1, bid_price_1, bid_size_1, ..., ..._L]
    """
    try:
        import pandas as pd  # Lazy import
    except ImportError as e:
        raise ImportError(
            "parse_orderbook_file requires pandas. Please install it: pip install pandas"
        ) from e

    # Read one small chunk to infer columns if needed
    preview = pd.read_csv(path, header=None, nrows=1)
    n_cols = preview.shape[1]

    inferred_levels = _infer_levels_from_orderbook_columns(n_cols)
    L = levels or inferred_levels
    if L != inferred_levels:
        # If user-provided levels disagree with file columns, raise to prevent silent misalignment
        raise ValueError(
            f"Provided levels={levels} does not match file columns={n_cols} (implies {inferred_levels} levels)."
        )

    names = orderbook_column_names(L)

    # Read full file with proper names; default dtype is fine (prices and sizes fit in 64-bit)
    df = pd.read_csv(path, header=None, names=names)
    return df


def iter_lobster(
    message_path: str,
    orderbook_path: str,
    *,
    chunksize: int = 100_000,
    time_as: str = "float",
) -> Generator[Tuple[Any, Any], None, None]:
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

    try:
        import pandas as pd  # Lazy import
    except ImportError as e:
        raise ImportError(
            "iter_lobster requires pandas. Please install it: pip install pandas"
        ) from e

    # Prepare readers
    msg_dtypes = (
        MESSAGE_DTYPES_FLOAT_TIME if time_as == "float" else MESSAGE_DTYPES_STR_TIME
    )

    msg_reader = pd.read_csv(
        message_path,
        header=None,
        names=MESSAGE_COLUMNS,
        dtype=msg_dtypes,
        chunksize=chunksize,
    )

    # Infer orderbook columns and prepare reader with names
    preview = pd.read_csv(orderbook_path, header=None, nrows=1)
    L = _infer_levels_from_orderbook_columns(preview.shape[1])
    ob_names = orderbook_column_names(L)

    ob_reader = pd.read_csv(
        orderbook_path,
        header=None,
        names=ob_names,
        chunksize=chunksize,
    )

    for msg_chunk, ob_chunk in zip(msg_reader, ob_reader):
        # Basic sanity check: ensure same number of rows in the chunk
        if len(msg_chunk) != len(ob_chunk):
            raise ValueError(
                f"Chunk misalignment: message rows {len(msg_chunk)} != orderbook rows {len(ob_chunk)}.\n"
                f"Ensure files correspond to the same ticker/day/level."
            )
        yield msg_chunk, ob_chunk


def read_lobster(
    message_path: str,
    orderbook_path: str,
    *,
    time_as: str = "float",
) -> Tuple[Any, Any]:
    """
    Convenience function to read the entire day into memory, returning (messages, orderbook).

    For large files prefer iter_lobster(...).
    """
    messages = parse_message_file(message_path, time_as=time_as)
    orderbook = parse_orderbook_file(orderbook_path)
    if len(messages) != len(orderbook):
        raise ValueError(
            f"Row count mismatch: message rows {len(messages)} != orderbook rows {len(orderbook)}.\n"
            f"Ensure files correspond to the same ticker/day/level."
        )
    return messages, orderbook


if __name__ == "__main__":
    # Minimal usage example on the included sample files.
    # Adjust paths/ticker as needed.
    aapl_dir = "LOBSTER_SampleFile_AAPL_2012-06-21_50"
    msg_path = f"{aapl_dir}/AAPL_2012-06-21_34200000_37800000_message_50.csv"
    ob_path = f"{aapl_dir}/AAPL_2012-06-21_34200000_37800000_orderbook_50.csv"
    try:
        msgs, obs = read_lobster(msg_path, ob_path)
        print("Loaded:", len(msgs), "events,", obs.shape[1] // 4, "levels")
        print(msgs.head())
        print(obs.head())
    except FileNotFoundError:
        # If run from a different CWD, suppress sample demo
        pass
