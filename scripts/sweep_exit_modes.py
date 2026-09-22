"""Compare experimental VCP exit models on the same portfolio setup.

Baseline entry rules, risk, costs, universe and period remain unchanged.
Only the exit mode varies.

Run:
    python scripts/sweep_exit_modes.py --start 2021-01-01 --end 2025-12-31
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vcp_strategy import PortfolioBacktester
from vcp_strategy.config import load_config


MODES = ["stop_only", "ma20", "ma50", "max126", "max252", "ma50_max252"]


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
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "baseline.yaml")
    parser.add_argument("--start", default="2021-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--capital", type=float, default=1_000_000.0)
    args = parser.parse_args()

    config = load_config(args.config)
    data = load_csvs(args.data_dir)

    rows = []
    for mode in MODES:
        bt = PortfolioBacktester(config, starting_capital=args.capital)
        result = bt.run(
            data,
            start=args.start,
            end=args.end,
            show_progress=False,
            exit_mode=mode,
        )
        metrics = bt.metrics(result, starting_capital=args.capital)
        rows.append({"exit_mode": mode, **metrics})

    report = pd.DataFrame(rows)
    cols = [
        "exit_mode",
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

    print("=" * 110)
    print("VCP EXIT-MODE SENSITIVITY")
    print("=" * 110)
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
    path = out / "exit_mode_sensitivity.csv"
    report.to_csv(path, index=False)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
