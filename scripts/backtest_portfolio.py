"""Run the VCP multi-stock portfolio backtest on CSV OHLCV data.

Expected data directory:
    data/
      RELIANCE.csv
      TCS.csv
      INFY.csv
      ...

Each CSV must contain:
    Date, Open, High, Low, Close, Volume

Primary research period:
    2015-01-01 -> 2025-12-31

Example:
    python scripts/backtest_portfolio.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow direct execution from the repository root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd

from vcp_strategy import PortfolioBacktester
from vcp_strategy.config import load_config


def load_csvs(data_dir: Path) -> dict[str, pd.DataFrame]:
    data: dict[str, pd.DataFrame] = {}

    for path in sorted(data_dir.glob("*.csv")):
        if path.name == "download_failures.txt":
            continue

        df = pd.read_csv(path)
        required = {"Date", "Open", "High", "Low", "Close", "Volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{path.name}: missing columns: {sorted(missing)}")

        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").drop_duplicates("Date").set_index("Date")
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()

        if not df.empty:
            data[path.stem] = df

    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "baseline.yaml",
    )
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--capital", type=float, default=1_000_000.0)
    args = parser.parse_args()

    if not args.data_dir.exists():
        raise SystemExit(
            f"Data directory not found: {args.data_dir}. "
            "Run scripts/download_data.py first."
        )

    data = load_csvs(args.data_dir)
    if not data:
        raise SystemExit("No usable OHLCV CSV files found.")

    config = load_config(args.config)
    backtester = PortfolioBacktester(config, starting_capital=args.capital)
    result = backtester.run(data, start=args.start, end=args.end)
    metrics = backtester.metrics(result, starting_capital=args.capital)

    print("=" * 60)
    print("VCP MULTI-STOCK PORTFOLIO BACKTEST")
    print("=" * 60)
    print(f"Symbols loaded:      {len(data)}")
    print(f"Backtest period:     {args.start} -> {args.end}")
    print(f"Initial capital:     ₹{args.capital:,.2f}")
    print(f"Max positions:       {config.risk.max_simultaneous_positions}")
    print(f"Risk per trade:      {config.risk.portfolio_risk_per_trade:.2%}")
    print("")
    print(f"Trades:              {metrics['trades']}")
    print(f"CAGR:                {metrics['cagr']:.2%}")
    print(f"Max drawdown:        {metrics['max_drawdown']:.2%}")
    print(f"Sharpe:              {metrics['sharpe']:.2f}")
    print(f"Win rate:            {metrics['win_rate']:.2%}")
    print(f"Profit factor:       {metrics['profit_factor']:.2f}")
    print(f"Average trade:       {metrics['average_trade_return']:.2%}")
    print(f"Average R:           {metrics['average_r_multiple']:.2f}")
    print(f"Ending capital:      ₹{metrics['ending_capital']:,.2f}")
    print(f"Net profit:          ₹{metrics['net_profit']:,.2f}")

    report_dir = PROJECT_ROOT / "reports"
    report_dir.mkdir(exist_ok=True)

    trades_path = report_dir / "portfolio_trades.csv"
    equity_path = report_dir / "portfolio_equity.csv"

    PortfolioBacktester.trades_frame(result).to_csv(trades_path, index=False)
    result.equity_curve.to_csv(equity_path, header=True)

    print("")
    print(f"Trades saved:        {trades_path}")
    print(f"Equity curve saved:  {equity_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
