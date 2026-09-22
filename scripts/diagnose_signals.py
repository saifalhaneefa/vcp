"""Diagnose VCP candidates and show exactly where signals are rejected.

Usage:
    python scripts/diagnose_signals.py
    python scripts/diagnose_signals.py --symbol ADANIGREEN
    python scripts/diagnose_signals.py --symbol ADANIGREEN --verbose
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
    keep_candidates: int = 25,
) -> tuple[dict, list[dict]]:
    df = add_indicators(raw, detector.cfg.trend)
    df["VolumeMA"] = df["Volume"].rolling(
        detector.cfg.vcp.volume_ma_days,
        min_periods=detector.cfg.vcp.volume_ma_days,
    ).mean()

    trend_ok = 0
    valid_pattern = 0
    volume_pass = 0
    pivot_pass = 0
    transition_pass = 0
    complete_signals = 0
    details: list[dict] = []

    for i in range(1, len(df) - 1):
        row = df.iloc[i]

        if not detector.trend_ok(row):
            continue
        trend_ok += 1

        start_idx = max(0, i - detector.cfg.vcp.max_contraction_lookback_days)
        setup = df.iloc[start_idx:i]
        contractions = detector.find_contractions(setup)
        if not detector.valid(contractions):
            continue
        valid_pattern += 1

        pivot = detector._pivot_before(df, i)
        ratio = float(row.Volume / row.VolumeMA) if row.VolumeMA > 0 else float("nan")
        close = float(row.Close)
        previous_close = float(df.iloc[i - 1].Close)

        volume_ok = bool(np.isfinite(ratio) and ratio >= detector.cfg.vcp.breakout_volume_multiple)
        pivot_ok = bool(np.isfinite(pivot) and close > pivot)
        transition_ok = bool(pivot_ok and previous_close < pivot)

        if volume_ok:
            volume_pass += 1
        if pivot_ok:
            pivot_pass += 1
        if transition_ok:
            transition_pass += 1

        if volume_ok and pivot_ok and transition_ok:
            complete_signals += 1
            next_date = pd.Timestamp(df.index[i + 1])
            raw_open = float(df.iloc[i + 1].Open)
            entry = raw_open * (1 + detector.cfg.costs.slippage_bps / 10000.0)
            max_loss_stop = entry * (1 - detector.cfg.risk.max_stop_loss_pct)
            pattern_stop = close * (1 - detector.cfg.risk.max_stop_loss_pct)
            stop = max(pattern_stop, max_loss_stop)

            entry_status = "eligible"
            if next_date < start or next_date > end:
                entry_status = "outside_backtest"

            details.append(
                {
                    "symbol": symbol,
                    "signal_date": pd.Timestamp(df.index[i]).date().isoformat(),
                    "entry_date": next_date.date().isoformat(),
                    "signal_close": close,
                    "pivot": pivot,
                    "close_minus_pivot_pct": (close / pivot - 1) if pivot else np.nan,
                    "previous_close": previous_close,
                    "breakout_volume_ratio": ratio,
                    "entry_open": raw_open,
                    "simulated_entry": entry,
                    "stop": stop,
                    "planned_loss_pct": 1 - stop / entry,
                    "entry_status": entry_status,
                }
            )

    summary = {
        "symbol": symbol,
        "rows": len(df),
        "trend_ok": trend_ok,
        "valid_vcp_pattern": valid_pattern,
        "volume_pass": volume_pass,
        "pivot_pass": pivot_pass,
        "transition_pass": transition_pass,
        "complete_signals": complete_signals,
    }
    return summary, details


def build_rejection_report(
    symbol: str,
    raw: pd.DataFrame,
    detector: VCPDetector,
) -> pd.DataFrame:
    df = add_indicators(raw, detector.cfg.trend)
    df["VolumeMA"] = df["Volume"].rolling(
        detector.cfg.vcp.volume_ma_days,
        min_periods=detector.cfg.vcp.volume_ma_days,
    ).mean()

    rows: list[dict] = []

    for i in range(1, len(df) - 1):
        row = df.iloc[i]
        if not detector.trend_ok(row):
            continue

        start_idx = max(0, i - detector.cfg.vcp.max_contraction_lookback_days)
        setup = df.iloc[start_idx:i]
        contractions = detector.find_contractions(setup)
        if not detector.valid(contractions):
            continue

        pivot = detector._pivot_before(df, i)
        ratio = float(row.Volume / row.VolumeMA) if row.VolumeMA > 0 else float("nan")
        close = float(row.Close)
        previous_close = float(df.iloc[i - 1].Close)

        volume_ok = np.isfinite(ratio) and ratio >= detector.cfg.vcp.breakout_volume_multiple
        pivot_ok = np.isfinite(pivot) and close > pivot
        transition_ok = np.isfinite(pivot) and previous_close < pivot

        failed: list[str] = []
        if not volume_ok:
            failed.append("volume")
        if not pivot_ok:
            failed.append("close_not_above_pivot")
        if pivot_ok and not transition_ok:
            failed.append("already_above_pivot")

        rows.append(
            {
                "symbol": symbol,
                "date": pd.Timestamp(df.index[i]).date().isoformat(),
                "close": close,
                "pivot": pivot,
                "close_minus_pivot_pct": (close / pivot - 1) if np.isfinite(pivot) else np.nan,
                "previous_close": previous_close,
                "breakout_volume_ratio": ratio,
                "failed_stage": ",".join(failed) if failed else "PASS",
            }
        )

    return pd.DataFrame(rows)


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
    parser.add_argument("--verbose", action="store_true")
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
    complete_details = []
    rejection_reports = []

    for path in paths:
        if path.name == "download_failures.txt" or not path.exists():
            continue

        raw = load_data(path)
        summary, details = diagnose(
            path.stem, raw, detector, start, end
        )
        summaries.append(summary)
        complete_details.extend(details)

        if args.verbose:
            rejection_reports.append(
                build_rejection_report(path.stem, raw, detector)
            )

    summary_report = pd.DataFrame(summaries)
    detail_report = pd.DataFrame(complete_details)

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

    if args.verbose and rejection_reports:
        verbose_report = pd.concat(rejection_reports, ignore_index=True)
        print("\nVALID-VCP REJECTION DETAILS")
        print(verbose_report.to_string(index=False))
        verbose_report.to_csv(out / "signal_rejections.csv", index=False)
        print(f"Saved: {out / 'signal_rejections.csv'}")


if __name__ == "__main__":
    main()
