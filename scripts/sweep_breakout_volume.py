"""Sensitivity analysis for breakout confirmation rules.

This script does not change the baseline configuration. It reports how many
complete VCP signals are produced when breakout-volume confirmation is varied.

Run:
    python scripts/sweep_breakout_volume.py
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

from vcp_strategy.config import load_config
from vcp_strategy.indicators import add_indicators
from vcp_strategy.signal import VCPDetector


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"Date", "Open", "High", "Low", "Close", "Volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns: {sorted(missing)}")
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date").drop_duplicates("Date").set_index("Date")


def count_signals(data, config, start, end) -> list[dict]:
    detector = VCPDetector(config)
    rows = []

    for symbol, raw in data.items():
        df = add_indicators(raw, config.trend)
        df["VolumeMA"] = df["Volume"].rolling(
            config.vcp.volume_ma_days,
            min_periods=config.vcp.volume_ma_days,
        ).mean()

        count = 0
        for i in range(1, len(df) - 1):
            date = pd.Timestamp(df.index[i])
            if date < start or date > end:
                continue
            if detector.find_signal(df, i) is not None:
                count += 1

        rows.append({"symbol": symbol, "complete_signals": count})

    return rows


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
    parser.add_argument(
        "--thresholds",
        default="1.0,1.25,1.5,1.75,2.0",
        help="Comma-separated breakout volume multiples.",
    )
    args = parser.parse_args()

    base = load_config(args.config)
    data = {
        path.stem: load_data(path)
        for path in sorted(args.data_dir.glob("*.csv"))
        if path.name != "download_failures.txt"
    }
    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    thresholds = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]

    report_rows = []
    for threshold in thresholds:
        cfg = replace(
            base,
            vcp=replace(base.vcp, breakout_volume_multiple=threshold),
        )
        rows = count_signals(data, cfg, start, end)
        total = sum(row["complete_signals"] for row in rows)
        active = sum(row["complete_signals"] > 0 for row in rows)
        report_rows.append(
            {
                "breakout_volume_multiple": threshold,
                "complete_signal_days": total,
                "symbols_with_signals": active,
            }
        )

    report = pd.DataFrame(report_rows)

    print("=" * 80)
    print("BREAKOUT VOLUME SENSITIVITY")
    print("=" * 80)
    print(f"Baseline transition lookback: {base.vcp.breakout_transition_lookback_days} day(s)")
    print(report.to_string(index=False))

    out = PROJECT_ROOT / "reports"
    out.mkdir(exist_ok=True)
    report.to_csv(out / "breakout_volume_sensitivity.csv", index=False)
    print(f"\nSaved: {out / 'breakout_volume_sensitivity.csv'}")


if __name__ == "__main__":
    main()
