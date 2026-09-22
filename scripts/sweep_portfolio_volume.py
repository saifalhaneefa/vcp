"""Sensitivity backtest for breakout-volume confirmation.

Runs the same portfolio engine at several breakout volume thresholds. No
baseline configuration is modified.

Run:
    python scripts/sweep_portfolio_volume.py
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vcp_strategy import PortfolioBacktester
from vcp_strategy.config import load_config


def load_csvs(data_dir: Path) -> dict[str, pd.DataFrame]:
    data = {}
    for path in sorted(data_dir.glob("*.csv")):
        if path.name == "download_failures.txt":
            continue
        df = pd.read_csv(path)
        required = {"Date", "Open", "High", "Low", "Close", "Volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{path.name}: missing columns: {sorted(missing)}")
        df["Date"] = pd.to_datetime(df["Date"])
        df = (
            df.sort_values("Date")
            .drop_duplicates("Date")
            .set_index("Date")
            [["Open", "High", "Low", "Close", "Volume"]]
            .dropna()
        )
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
    parser.add_argument(
        "--thresholds",
        default="1.0,1.25,1.5,1.75,2.0",
        help="Comma-separated breakout volume multiples.",
    )
    args = parser.parse_args()

    base = load_config(args.config)
    data = load_csvs(args.data_dir)
    thresholds = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]

    rows = []
    for threshold in thresholds:
        cfg = replace(
            base,
            vcp=replace(base.vcp, breakout_volume_multiple=threshold),
        )
        bt = PortfolioBacktester(cfg, starting_capital=args.capital)
        result = bt.run(data, start=args.start, end=args.end, show_progress=False)
        metrics = bt.metrics(result, starting_capital=args.capital)
        rows.append(
            {
                "breakout_volume_multiple": threshold,
                **metrics,
            }
        )

    report = pd.DataFrame(rows)
    cols = [
        "breakout_volume_multiple",
        "trades",
        "cagr",
        "max_drawdown",
        "sharpe",
        "win_rate",
        "profit_factor",
        "average_trade_return",
        "average_r_multiple",
        "ending_capital",
        "net_profit",
    ]
    report = report[cols]

    print("=" * 100)
    print("PORTFOLIO BREAKOUT VOLUME SENSITIVITY")
    print("=" * 100)
    print(f"Period: {args.start} -> {args.end}")
    print(f"Symbols: {len(data)}")
    print(report.to_string(index=False, formatters={
        "cagr": "{:.2%}".format,
        "max_drawdown": "{:.2%}".format,
        "sharpe": "{:.2f}".format,
        "win_rate": "{:.2%}".format,
        "profit_factor": "{:.2f}".format,
        "average_trade_return": "{:.2%}".format,
        "average_r_multiple": "{:.2f}".format,
        "ending_capital": "₹{:,.2f}".format,
        "net_profit": "₹{:,.2f}".format,
    }))

    out = PROJECT_ROOT / "reports"
    out.mkdir(exist_ok=True)
    path = out / "portfolio_volume_sensitivity.csv"
    report.to_csv(path, index=False)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
