"""Diagnose VCP candidates and portfolio-entry behavior.

Usage:
    python scripts/diagnose_signals.py
    python scripts/diagnose_signals.py --symbol ADANIGREEN
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
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


def diagnose(
    symbol: str,
    raw: pd.DataFrame,
    detector: VCPDetector,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[dict, list[dict]]:
    df = add_indicators(raw, detector.cfg.trend)
    df["VolumeMA"] = df["Volume"].rolling(
        detector.cfg.vcp.volume_ma_days,
        min_periods=detector.cfg.vcp.volume_ma_days,
    ).mean()

    trend_ok = 0
    valid_pattern = 0
    breakout_confirmed = 0
    complete_signals = 0
    details: list[dict] = []

    for i in range(1, len(df) - 1):
        if not detector.trend_ok(df.iloc[i]):
            continue
        trend_ok += 1

        start_idx = max(0, i - detector.cfg.vcp.max_contraction_lookback_days)
        setup = df.iloc[start_idx:i]
        contractions = detector.find_contractions(setup)
        if not detector.valid(contractions):
            continue
        valid_pattern += 1

        pivot = detector._pivot_before(df, i)
        row = df.iloc[i]
        ratio = float(row.Volume / row.VolumeMA) if row.VolumeMA > 0 else float("nan")

        if not pd.notna(ratio) or ratio < detector.cfg.vcp.breakout_volume_multiple:
            continue
        if row.Close <= pivot:
            continue

        previous_pivot = detector._pivot_before(df, i - 1)
        previous_close = float(df.iloc[i - 1].Close)
        if np.isfinite(previous_pivot) and previous_close >= previous_pivot:
            continue

        breakout_confirmed += 1

        signal = detector.find_signal(df, i)
        if signal is None:
            continue

        complete_signals += 1
        signal_date = pd.Timestamp(df.index[i])
        next_date = pd.Timestamp(df.index[i + 1])
        raw_open = float(df.iloc[i + 1].Open)
        entry = raw_open * (
            1.0 + detector.cfg.costs.slippage_bps / 10000.0
        )
        max_loss_stop = entry * (1.0 - detector.cfg.risk.max_stop_loss_pct)
        pattern_stop = float(signal.stop_reference)
        stop = max(pattern_stop, max_loss_stop)
        per_share_risk = entry - stop

        entry_status = "eligible"
        if next_date < start or next_date > end:
            entry_status = "outside_backtest"
        elif per_share_risk <= 0:
            entry_status = "skipped_gap_through_stop"

        details.append(
            {
                "symbol": symbol,
                "signal_date": signal_date.date().isoformat(),
                "entry_date": next_date.date().isoformat(),
                "signal_close": float(row.Close),
                "pivot": pivot,
                "breakout_volume_ratio": ratio,
                "entry_open": raw_open,
                "simulated_entry": entry,
                "stop": stop,
                "planned_loss_pct": 1.0 - stop / entry,
                "per_share_risk": per_share_risk,
                "entry_status": entry_status,
            }
        )

    summary = {
        "symbol": symbol,
        "rows": len(df),
        "trend_ok": trend_ok,
        "valid_vcp_pattern": valid_pattern,
        "breakout_confirmed": breakout_confirmed,
        "complete_signals": complete_signals,
    }
    return summary, details


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "baseline.yaml",
    )
    parser.add_argument("--symbol", default=None)
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2025-12-31")
    args = parser.parse_args()

    config = load_config(args.config)
    detector = VCPDetector(config)
    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)

    paths = (
        [args.data_dir / f"{args.symbol}.csv"]
        if args.symbol
        else sorted(args.data_dir.glob("*.csv"))
    )

    summaries = []
    details = []

    for path in paths:
        if path.name == "download_failures.txt" or not path.exists():
            continue
        summary, rows = diagnose(
            path.stem, load_data(path), detector, start, end
        )
        summaries.append(summary)
        details.extend(rows)

    summary_report = pd.DataFrame(summaries)
    detail_report = pd.DataFrame(details)

    if summary_report.empty:
        raise SystemExit("No usable CSV files found.")

    print("SIGNAL SUMMARY")
    print(summary_report.to_string(index=False))

    if not detail_report.empty:
        print("\nCOMPLETE SIGNALS")
        print(detail_report.to_string(index=False))

    out = PROJECT_ROOT / "reports"
    out.mkdir(exist_ok=True)
    summary_report.to_csv(out / "signal_diagnostics.csv", index=False)
    detail_report.to_csv(out / "signal_candidates.csv", index=False)

    print(f"\nSaved: {out / 'signal_diagnostics.csv'}")
    print(f"Saved: {out / 'signal_candidates.csv'}")


if __name__ == "__main__":
    main()
