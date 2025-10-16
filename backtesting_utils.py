from typing import Any, Optional, Tuple

def _get_value(row: Any, col: str) -> Any:
    if hasattr(row, "get"):
        return row[col]
    else:
        return row[col]

def _side_and_exec(message_row: Any) -> Optional[str]:
    mtype = int(_get_value(message_row, "type"))
    if mtype not in (4, 5):
        return None
    direction = int(_get_value(message_row, "direction"))
    if direction == 1:
        return "bid"
    elif direction == -1:
        return "ask"
    return None

def _best_quotes(orderbook_row: Any) -> Tuple[int, int, int, int]:
    ask_px = int(_get_value(orderbook_row, "ask_price_1"))
    ask_sz = int(_get_value(orderbook_row, "ask_size_1"))
    bid_px = int(_get_value(orderbook_row, "bid_price_1"))
    bid_sz = int(_get_value(orderbook_row, "bid_size_1"))
    return ask_px, ask_sz, bid_px, bid_sz
