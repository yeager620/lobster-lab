from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any

from lobster_parsing import iter_lobster
from backtesting_utils import _get_value, _side_and_exec, _best_quotes


@dataclass
class MarketMakerParams:
    quote_size: int = 100
    participation: float = 0.1
    inventory_limit: int = 1_000
    fee_per_share: float = 0.0
    impact_bps: float = 0.0


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


def run_backtest_for_ticker(
    ticker: str,
    message_path: str,
    orderbook_path: str,
    params: Optional[MarketMakerParams] = None,
) -> BacktestResult:
    params = params or MarketMakerParams()

    inv = 0
    cash_ticks = 0
    trades = 0
    shares_bought = 0
    shares_sold = 0
    notional_bought_ticks = 0
    notional_sold_ticks = 0

    last_mid_ticks = 0
    events = 0

    for msg_chunk, ob_chunk in iter_lobster(
        message_path, orderbook_path, chunksize=200_000, time_as="float"
    ):
        for i in range(len(msg_chunk)):
            events += 1
            msg = msg_chunk[i]
            ob = ob_chunk[i]

            ask_px, ask_sz, bid_px, bid_sz = _best_quotes(ob)

            if bid_px > 0 and ask_px < 9_999_999_999 and bid_px <= ask_px:
                last_mid_ticks = (bid_px + ask_px) // 2

            side = _side_and_exec(msg)
            if side is None:
                continue

            exec_price = int(_get_value(msg, "price"))
            exec_size = int(_get_value(msg, "size"))
            if exec_price <= 0 or exec_size <= 0:
                continue

            want_bid = inv < params.inventory_limit
            want_ask = inv > -params.inventory_limit

            if side == "bid":
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
                        if params.impact_bps:
                            impact_cost = int(
                                round(notional * (params.impact_bps / 10000.0))
                            )
                            if impact_cost > 0:
                                cash_ticks -= impact_cost
            elif side == "ask":
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
                        if params.impact_bps:
                            impact_cost = int(
                                round(notional * (params.impact_bps / 10000.0))
                            )
                            if impact_cost > 0:
                                cash_ticks -= impact_cost

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
    params = params or MarketMakerParams()
    results: List[BacktestResult] = []

    for t in tickers:
        if t not in data_roots:
            raise ValueError(f"Missing data paths for ticker {t}")
        msg_path, ob_path = data_roots[t]
        res = run_backtest_for_ticker(t, msg_path, ob_path, params)
        results.append(res)

    agg = {
        "tickers": tickers,
        "events": sum(r.events for r in results),
        "trades": sum(r.trades for r in results),
        "shares_bought": sum(r.shares_bought for r in results),
        "shares_sold": sum(r.shares_sold for r in results),
        "end_inventory": sum(r.end_inventory for r in results),
        "cash_usd": sum(r.cash_ticks for r in results) / 10000.0,
        "realized_usd": sum(r.realized_ticks for r in results) / 10000.0,
        "mtm_usd": sum(r.mtm_ticks for r in results) / 10000.0,
        "total_pnl_usd": sum(r.total_pnl_ticks for r in results) / 10000.0,
    }
    return results, agg
