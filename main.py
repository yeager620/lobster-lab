import argparse

from backtesting import MarketMakerParams, run_backtest_multiple


SAMPLE_FILES = {
    "AAPL": (
        "LOBSTER_SampleFile_AAPL_ 2012-06-21_50/AAPL_2012-06-21_34200000_37800000_message_50.csv",
        "LOBSTER_SampleFile_AAPL_2012-06-21_50/AAPL_2012-06-21_34200000_37800000_orderbook_50.csv",
    ),
    "MSFT": (
        "LOBSTER_SampleFile_MSFT_2012-06-21_50/MSFT_2012-06-21_34200000_37800000_message_50.csv",
        "LOBSTER_SampleFile_MSFT_2012-06-21_50/MSFT_2012-06-21_34200000_37800000_orderbook_50.csv",
    ),
    "SPY": (
        "LOBSTER_SampleFile_SPY_2012-06-21_50/SPY_2012-06-21_34200000_37800000_message_50.csv",
        "LOBSTER_SampleFile_SPY_2012-06-21_50/SPY_2012-06-21_34200000_37800000_orderbook_50.csv",
    ),
    "GOOG": (
        "LOBSTER_SampleFile_GOOG_2012-06-21_10/GOOG_2012-06-21_34200000_57600000_message_10.csv",
        "LOBSTER_SampleFile_GOOG_2012-06-21_10/GOOG_2012-06-21_34200000_57600000_orderbook_10.csv",
    ),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LOBSTER market making backtest (simple baseline)"
    )
    p.add_argument(
        "--tickers",
        nargs="+",
        default=["AAPL"],
        help="1 to 4 tickers from: AAPL MSFT SPY GOOG",
    )
    p.add_argument(
        "--quote-size", type=int, default=100, help="Quote size per side (shares)"
    )
    p.add_argument(
        "--participation",
        type=float,
        default=0.1,
        help="Participation rate of best-quote executions [0..1]",
    )
    p.add_argument(
        "--inv-limit",
        type=int,
        default=1000,
        help="Inventory limit in shares per ticker",
    )
    p.add_argument(
        "--fee-per-share",
        type=float,
        default=0.0,
        help="Fee per share in USD (e.g., 0.0005)",
    )
    p.add_argument(
        "--impact-bps",
        type=float,
        default=0.0,
        help="Temporary market impact in basis points of price applied to each fill (slippage)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tickers = [t.upper() for t in args.tickers]
    if not (1 <= len(tickers) <= 4):
        raise SystemExit("Please specify between 1 and 4 tickers")

    missing = [t for t in tickers if t not in SAMPLE_FILES]
    if missing:
        raise SystemExit(
            f"Missing sample data for tickers: {missing}. Available: {list(SAMPLE_FILES)}"
        )

    params = MarketMakerParams(
        quote_size=args.quote_size,
        participation=args.participation,
        inventory_limit=args.inv_limit,
        fee_per_share=args.fee_per_share,
        impact_bps=args.impact_bps,
    )

    data_map = {t: SAMPLE_FILES[t] for t in tickers}

    results, agg = run_backtest_multiple(tickers, data_map, params)

    def fmt_usd(x: float) -> str:
        return f"${x:,.2f}"

    print("Backtest summary:")
    for r in results:
        dollars = r.to_dollars()
        print(
            f"- {r.ticker}: events={r.events:,}, trades={r.trades:,}, end_inv={r.end_inventory:,} sh, "
            f"realized={fmt_usd(dollars['realized_usd'])}, mtm={fmt_usd(dollars['mtm_usd'])}, "
            f"total={fmt_usd(dollars['total_pnl_usd'])}"
        )
    print(
        f"Aggregate over {len(results)} tickers: trades={agg['trades']:,}, end_inv={agg['end_inventory']:,} sh, "
        f"realized={fmt_usd(agg['realized_usd'])}, mtm={fmt_usd(agg['mtm_usd'])}, total={fmt_usd(agg['total_pnl_usd'])}"
    )


if __name__ == "__main__":
    main()
