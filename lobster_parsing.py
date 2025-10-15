from __future__ import annotations

from typing import Generator, List, Optional, Tuple, Any
import polars as pl


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


def parse_message_file(path: str, time_as: str = "float") -> pl.DataFrame:
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
    preview = pl.read_csv(path, has_header=False, n_rows=1)
    n_cols = len(preview.columns)

    inferred_levels = _infer_levels_from_orderbook_columns(n_cols)
    L = levels or inferred_levels
    if L != inferred_levels:
        raise ValueError(
            f"Provided levels={levels} does not match file columns={n_cols} (implies {inferred_levels} levels)."
        )

    names = orderbook_column_names(L)

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
    if time_as not in ("float", "str"):
        raise ValueError("time_as must be 'float' or 'str'")

    schema = MESSAGE_SCHEMA.copy()
    if time_as == "str":
        schema["time"] = pl.Utf8

    ob_preview = pl.read_csv(orderbook_path, has_header=False, n_rows=1)
    L = _infer_levels_from_orderbook_columns(len(ob_preview.columns))
    ob_names = orderbook_column_names(L)

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
    aapl_dir = "LOBSTER_SampleFile_AAPL_2012-06-21_50"
    msg_path = f"{aapl_dir}/AAPL_2012-06-21_34200000_37800000_message_50.csv"
    ob_path = f"{aapl_dir}/AAPL_2012-06-21_34200000_37800000_orderbook_50.csv"
    try:
        msgs, obs = read_lobster(msg_path, ob_path)
        print("Loaded:", len(msgs), "events,", len(obs.columns) // 4, "levels")
        print(msgs.head())
        print(obs.head())
    except FileNotFoundError:
        pass
