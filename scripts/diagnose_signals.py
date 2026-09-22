"""Diagnose where VCP candidates are being rejected.

Usage:
    python scripts/diagnose_signals.py
    python scripts/diagnose_signals.py --symbol ADANIGREEN
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


def diagnose(symbol: str, raw: pd.DataFrame, detector: VCPDetector) -> dict:
    df = add_indicators(raw, detector.cfg.trend)
    df["VolumeMA"] = df["Volume"].rolling(
        detector.cfg.vcp.volume_ma_days,
        min_periods=detector.cfg.vcp.volume_ma_days,
    ).mean()

    trend_ok = 0
    valid_pattern = 0
    volume_breakout = 0
    complete_signals = 0

    for i in range(1, len(df) - 1):
        if not detector.trend_ok(df.iloc[i]):
            continue
        trend_ok += 1

        start = max(0, i - detector.cfg.vcp.max_contraction_lookback_days)
        setup = df.iloc[start:i]
        contractions = detector.find_contractions(setup)
        if not detector.valid(contractions):
            continue
        valid_pattern += 1

        pivot = float(
            df.iloc[
                max(start, i - detector.cfg.vcp.pivot_lookback_days) : i
            ].High.max()
        )
        row = df.iloc[i]
        ratio = float(row.Volume / row.VolumeMA) if row.VolumeMA > 0 else float("nan")

        if not pd.notna(ratio) or ratio < detector.cfg.vcp.breakout_volume_multiple:
            continue
        if row.Close <= pivot:
            continue
        volume_breakout += 1

        if detector.find_signal(df, i) is not None:
            complete_signals += 1

    return {
        "symbol": symbol,
        "rows": len(df),
        "trend_ok": trend_ok,
        "valid_vcp_pattern": valid_pattern,
        "breakout_confirmed": volume_breakout,
        "complete_signals": complete_signals,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "baseline.yaml")
    parser.add_argument("--symbol", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    detector = VCPDetector(config)

    paths = (
        [args.data_dir / f"{args.symbol}.csv"]
        if args.symbol
        else sorted(args.data_dir.glob("*.csv"))
    )

    results = []
    for path in paths:
        if path.name == "download_failures.txt" or not path.exists():
            continue
        symbol = path.stem
        results.append(diagnose(symbol, load_data(path), detector))

    report = pd.DataFrame(results)
    if report.empty:
        raise SystemExit("No usable CSV files found.")

    print(report.to_string(index=False))

    out = PROJECT_ROOT / "reports"
    out.mkdir(exist_ok=True)
    report.to_csv(out / "signal_diagnostics.csv", index=False)
    print(f"\nSaved: {out / 'signal_diagnostics.csv'}")


if __name__ == "__main__":
    main()
