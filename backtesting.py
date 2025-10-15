"""
Simple market making backtesting engine over LOBSTER data.

Assumptions and model:
- We simulate a passive market maker that continuously joins the best bid and best ask with a fixed quote size.
- Fills occur only when there is an execution event in LOBSTER messages (Type 4 visible, Type 5 hidden).
- Side mapping per LOBSTER spec:
    * direction == 1 (buy limit order executed) -> trade on the bid side (seller-initiated). Our bid can be hit.
    * direction == -1 (sell limit order executed) -> trade on the ask side (buyer-initiated). Our ask can be lifted.
- Execution price must be positive; we require it to match the contemporaneous best price for a fill to be possible.
- Fill quantity is min(quote_size, floor(participation * exec_size)). If that result is 0, no fill.
- Inventory limits: when inv >= limit, we stop bidding; when inv <= -limit, we stop offering.
- PnL: realized cash from fills plus mark-to-market on residual inventory at the last midprice of the day.
- Prices are denominated in price ticks equal to dollars * 10000, consistent with LOBSTER CSVs.

This is a deliberately simple baseline for research and can be extended with queue models,
latency, skewed quotes, dynamic sizing, etc.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any

from lobster_parsing import iter_lobster


@dataclass
class MarketMakerParams:
    quote_size: int = 100  # shares per side
    participation: float = (
        0.1  # fraction of external marketable flow you capture at best price
    )
    inventory_limit: int = 1_000  # max absolute shares per ticker
    fee_per_share: float = (
        0.0  # in dollars per share; positive reduces PnL per share traded
    )
    impact_bps: float = 0.0  # temporary market impact in basis points of price, applied as slippage on each fill


@dataclass
class BacktestResult:
    ticker: str
    events: int
    trades: int
    shares_bought: int
    shares_sold: int
    end_inventory: int
    notional_bought_ticks: int
    notional_sold_ticks: int
    cash_ticks: int
    realized_ticks: int
    mtm_ticks: int
    total_pnl_ticks: int
    last_mid_ticks: int

    def to_dollars(self) -> Dict[str, Any]:
        # Convert tick-based values (price * shares with price in dollars*10000) back to dollars
        def ticks_to_dollars(x: int) -> float:
            return x / 10000.0

        d = asdict(self)
        for k in [
            "notional_bought_ticks",
            "notional_sold_ticks",
            "cash_ticks",
            "realized_ticks",
            "mtm_ticks",
            "total_pnl_ticks",
            "last_mid_ticks",
        ]:
            d[k.replace("_ticks", "_usd")] = ticks_to_dollars(d.pop(k))
        return d


def _side_and_exec(message_row: Any) -> Optional[str]:
    """Return 'bid' if a bid-side (sell-initiated) execution, 'ask' if ask-side (buy-initiated), else None."""
    mtype = int(message_row["type"])  # 4 visible exec, 5 hidden exec
    if mtype not in (4, 5):
        return None
    direction = int(
        message_row["direction"]
    )  # 1 buy limit order executed -> bid side; -1 sell limit -> ask side
    if direction == 1:
        return "bid"
    elif direction == -1:
        return "ask"
    return None


def _best_quotes(orderbook_row: Any) -> Tuple[int, int, int, int]:
    """Return (best_ask_px, best_ask_sz, best_bid_px, best_bid_sz) as integers (ticks, shares)."""
    ask_px = int(orderbook_row["ask_price_1"])  # dollars*10000
    ask_sz = int(orderbook_row["ask_size_1"])  # shares
    bid_px = int(orderbook_row["bid_price_1"])  # dollars*10000
    bid_sz = int(orderbook_row["bid_size_1"])  # shares
    return ask_px, ask_sz, bid_px, bid_sz


def run_backtest_for_ticker(
    ticker: str,
    message_path: str,
    orderbook_path: str,
    params: Optional[MarketMakerParams] = None,
) -> BacktestResult:
    params = params or MarketMakerParams()

    inv = 0  # inventory shares, +long
    cash_ticks = 0  # cash in ticks (dollars*10000)
    trades = 0
    shares_bought = 0
    shares_sold = 0
    notional_bought_ticks = 0
    notional_sold_ticks = 0

    last_mid_ticks = 0
    events = 0

    # Iterate streams in chunks to control memory
    for msg_chunk, ob_chunk in iter_lobster(
        message_path, orderbook_path, chunksize=200_000, time_as="float"
    ):
        # Iterate row-wise efficiently
        for i in range(len(msg_chunk)):
            events += 1
            msg = msg_chunk.iloc[i]
            ob = ob_chunk.iloc[i]

            ask_px, ask_sz, bid_px, bid_sz = _best_quotes(ob)
            # Compute mid for MTM updates later
            # For dummy price levels, mid is meaningless, but LOBSTER uses huge sentinels; keep last valid mid
            if bid_px > 0 and ask_px < 9_999_999_999 and bid_px <= ask_px:
                last_mid_ticks = (bid_px + ask_px) // 2

            # Determine if there's an execution event and on which side
            side = _side_and_exec(msg)
            if side is None:
                continue

            exec_price = int(msg["price"])  # ticks
            exec_size = int(msg["size"])  # shares
            if exec_price <= 0 or exec_size <= 0:
                continue

            # Our quoting logic: join best on both sides unless inventory limit reached
            want_bid = inv < params.inventory_limit
            want_ask = inv > -params.inventory_limit

            if side == "bid":
                # Seller-initiated trade hits bids at exec_price; fill if we are at best bid and want to bid
                if want_bid and exec_price == bid_px and bid_px > 0:
                    fill = int(params.participation * exec_size)
                    if fill > params.quote_size:
                        fill = params.quote_size
                    if fill > 0:
                        trades += 1
                        inv += fill
                        shares_bought += fill
                        notional = fill * bid_px
                        notional_bought_ticks += notional
                        cash_ticks -= notional
                        # Apply temporary market impact (slippage) as cost in ticks
                        if params.impact_bps:
                            impact_cost = int(
                                round(notional * (params.impact_bps / 10000.0))
                            )
                            if impact_cost > 0:
                                cash_ticks -= impact_cost
            elif side == "ask":
                # Buyer-initiated trade lifts offers at exec_price; fill if we are at best ask and want to offer
                if want_ask and exec_price == ask_px and ask_px > 0:
                    fill = int(params.participation * exec_size)
                    if fill > params.quote_size:
                        fill = params.quote_size
                    if fill > 0:
                        trades += 1
                        inv -= fill
                        shares_sold += fill
                        notional = fill * ask_px
                        notional_sold_ticks += notional
                        cash_ticks += notional
                        # Apply temporary market impact (slippage) as cost in ticks
                        if params.impact_bps:
                            impact_cost = int(
                                round(notional * (params.impact_bps / 10000.0))
                            )
                            if impact_cost > 0:
                                cash_ticks -= impact_cost

    # Apply fees (in dollars per share) to total traded shares
    total_shares_traded = shares_bought + shares_sold
    fee_ticks_per_share = int(round(params.fee_per_share * 10000))
    fees_ticks = total_shares_traded * fee_ticks_per_share
    cash_ticks -= fees_ticks

    realized_ticks = cash_ticks
    mtm_ticks = inv * last_mid_ticks
    total_pnl_ticks = realized_ticks + mtm_ticks

    return BacktestResult(
        ticker=ticker,
        events=events,
        trades=trades,
        shares_bought=shares_bought,
        shares_sold=shares_sold,
        end_inventory=inv,
        notional_bought_ticks=notional_bought_ticks,
        notional_sold_ticks=notional_sold_ticks,
        cash_ticks=cash_ticks,
        realized_ticks=realized_ticks,
        mtm_ticks=mtm_ticks,
        total_pnl_ticks=total_pnl_ticks,
        last_mid_ticks=last_mid_ticks,
    )


def run_backtest_multiple(
    tickers: List[str],
    data_roots: Dict[str, Tuple[str, str]],
    params: Optional[MarketMakerParams] = None,
) -> Tuple[List[BacktestResult], Dict[str, Any]]:
    """
    Run backtest sequentially over 1..4 tickers and return per-ticker results and aggregate metrics.

    - tickers: list of tickers to include.
    - data_roots: mapping ticker -> (message_path, orderbook_path).
    """
    params = params or MarketMakerParams()
    results: List[BacktestResult] = []

    for t in tickers:
        if t not in data_roots:
            raise ValueError(f"Missing data paths for ticker {t}")
        msg_path, ob_path = data_roots[t]
        res = run_backtest_for_ticker(t, msg_path, ob_path, params)
        results.append(res)

    # Aggregate
    agg = {
        "tickers": tickers,
        "events": sum(r.events for r in results),
        "trades": sum(r.trades for r in results),
        "shares_bought": sum(r.shares_bought for r in results),
        "shares_sold": sum(r.shares_sold for r in results),
        "end_inventory": sum(r.end_inventory for r in results),
        "cash_usd": sum(r.cash_ticks for r in results) / 10000.0,
        "realized_usd": sum(r.realized_ticks for r in results) / 10000.0,
        # MTM and total PnL aggregate use each ticker's last mid for inventory valuation; add across
        "mtm_usd": sum(r.mtm_ticks for r in results) / 10000.0,
        "total_pnl_usd": sum(r.total_pnl_ticks for r in results) / 10000.0,
    }
    return results, agg
