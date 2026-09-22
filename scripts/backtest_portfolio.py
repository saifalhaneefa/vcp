"""Run the VCP portfolio backtest on CSV OHLCV data.

Expected data directory:
    data/
      RELIANCE.csv
      TCS.csv
      INFY.csv
      ...

Each CSV must contain:
    Date, Open, High, Low, Close, Volume

Example:
    python scripts/backtest_portfolio.py --data-dir data --start 2015-01-01 --end 2025-12-31
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from vcp_strategy.backtest import Backtester
from vcp_strategy.config import load_config


def load_csvs(data_dir: Path, start: str, end: str) -> dict[str, pd.DataFrame]:
    data: dict[str, pd.DataFrame] = {}

    for path in sorted(data_dir.glob("*.csv")):
        df = pd.read_csv(path)
        required = {"Date", "Open", "High", "Low", "Close", "Volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{path.name}: missing columns: {sorted(missing)}")

        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").drop_duplicates("Date").set_index("Date")
        df = df.loc[start:end, ["Open", "High", "Low", "Close", "Volume"]].dropna()

        if not df.empty:
            data[path.stem] = df

    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2025-12-31")
    args = parser.parse_args()

    if not args.data_dir.exists():
        raise SystemExit(
            f"Data directory not found: {args.data_dir}. "
            "Put one OHLCV CSV per stock in this directory."
        )

    data = load_csvs(args.data_dir, args.start, args.end)
    if not data:
        raise SystemExit("No usable CSV files found.")

    print(f"Loaded {len(data)} symbols")
    print(f"Requested period: {args.start} -> {args.end}")
    print("")

    # V1 currently provides a per-symbol engine. This runner intentionally
    # exposes that limitation rather than pretending it is a portfolio test.
    config = load_config(args.config)
    backtester = Backtester(config)

    results = []
    for symbol, df in data.items():
        trades = backtester.run_symbol(symbol, df)
        metrics = backtester.metrics(trades)
        metrics["symbol"] = symbol
        results.append(metrics)

    report = pd.DataFrame(results).sort_values("symbol")
    print(report.to_string(index=False))

    output_dir = Path("reports")
    output_dir.mkdir(exist_ok=True)
    report.to_csv(output_dir / "v1_symbol_results.csv", index=False)
    print(f"\nSaved: {output_dir / 'v1_symbol_results.csv'}")
    print(
        "\nNOTE: This is NOT the final multi-stock portfolio backtest. "
        "V2 will manage shared cash, simultaneous positions, portfolio risk, "
        "and a single portfolio equity curve."
    )


if __name__ == "__main__":
    main()
