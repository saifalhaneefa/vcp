"""Diagnose VCP candidates and show exactly where signals are rejected."""
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


def diagnose(symbol, raw, detector, start, end):
    df = add_indicators(raw, detector.cfg.trend)
    df["VolumeMA"] = df["Volume"].rolling(
        detector.cfg.vcp.volume_ma_days,
        min_periods=detector.cfg.vcp.volume_ma_days,
    ).mean()

    counts = {
        "trend_ok": 0,
        "valid_vcp_pattern": 0,
        "fresh_contraction": 0,
        "runup_ok": 0,
        "volume_pass": 0,
        "pivot_pass": 0,
        "transition_pass": 0,
        "extension_ok": 0,
        "complete_signals": 0,
    }
    details = []
    candidate_debug = []

    for i in range(1, len(df) - 1):
        row = df.iloc[i]

        if not detector.trend_ok(row):
            continue
        counts["trend_ok"] += 1

        start_idx = max(0, i - detector.cfg.vcp.max_contraction_lookback_days)
        setup = df.iloc[start_idx:i]
        cs = detector.find_contractions(setup)
        if not detector.valid(cs):
            continue
        counts["valid_vcp_pattern"] += 1

        pivot, final_age, runup = detector._breakout_context(df, i, cs)
        fresh_ok = (
            np.isfinite(final_age)
            and final_age <= detector.cfg.vcp.max_final_contraction_age_bars
        )
        runup_ok = (
            np.isfinite(runup)
            and runup <= detector.cfg.vcp.max_post_contraction_runup
        )
        if fresh_ok:
            counts["fresh_contraction"] += 1
        if runup_ok:
            counts["runup_ok"] += 1

        ratio = (
            float(row.Volume / row.VolumeMA)
            if row.VolumeMA > 0
            else float("nan")
        )
        close = float(row.Close)
        previous_close = float(df.iloc[i - 1].Close)

        volume_ok = (
            np.isfinite(ratio)
            and ratio >= detector.cfg.vcp.breakout_volume_multiple
        )
        pivot_ok = np.isfinite(pivot) and close > pivot
        transition_ok = np.isfinite(pivot) and previous_close <= pivot
        extension_ok = (
            pivot_ok
            and close / pivot - 1.0 <= detector.cfg.vcp.max_breakout_extension
        )

        if volume_ok:
            counts["volume_pass"] += 1
        if pivot_ok:
            counts["pivot_pass"] += 1
        if transition_ok:
            counts["transition_pass"] += 1
        if extension_ok:
            counts["extension_ok"] += 1

        final = cs[-1]
        candidate_debug.append(
            {
                "date": pd.Timestamp(df.index[i]).date().isoformat(),
                "final_high_date": pd.Timestamp(
                    setup.index[final.high_idx]
                ).date().isoformat(),
                "final_low_date": pd.Timestamp(
                    setup.index[final.low_idx]
                ).date().isoformat(),
                "final_high": final.high,
                "final_low": final.low,
                "final_depth_pct": final.depth,
                "final_age_bars": final_age,
                "post_contraction_runup_pct": runup,
                "pivot": pivot,
                "close": close,
                "close_minus_pivot_pct": (
                    close / pivot - 1
                    if np.isfinite(pivot)
                    else np.nan
                ),
                "previous_close": previous_close,
                "volume_ratio": ratio,
                "volume_ok": volume_ok,
                "pivot_ok": pivot_ok,
                "transition_ok": transition_ok,
                "extension_ok": extension_ok,
                "fresh_ok": fresh_ok,
                "runup_ok": runup_ok,
                "complete": (
                    fresh_ok
                    and runup_ok
                    and volume_ok
                    and pivot_ok
                    and transition_ok
                    and extension_ok
                ),
            }
        )

        if (
            fresh_ok
            and runup_ok
            and volume_ok
            and pivot_ok
            and transition_ok
            and extension_ok
        ):
            counts["complete_signals"] += 1
            next_date = pd.Timestamp(df.index[i + 1])
            raw_open = float(df.iloc[i + 1].Open)
            entry = raw_open * (
                1 + detector.cfg.costs.slippage_bps / 10000.0
            )
            stop = max(
                close * (1 - detector.cfg.risk.max_stop_loss_pct),
                entry * (1 - detector.cfg.risk.max_stop_loss_pct),
            )
            details.append(
                {
                    "symbol": symbol,
                    "signal_date": pd.Timestamp(df.index[i]).date().isoformat(),
                    "entry_date": next_date.date().isoformat(),
                    "signal_close": close,
                    "pivot": pivot,
                    "close_minus_pivot_pct": close / pivot - 1,
                    "final_contraction_age_bars": final_age,
                    "post_contraction_runup": runup,
                    "breakout_volume_ratio": ratio,
                    "entry_open": raw_open,
                    "simulated_entry": entry,
                    "stop": stop,
                    "planned_loss_pct": 1 - stop / entry,
                    "entry_status": (
                        "eligible"
                        if start <= next_date <= end
                        else "outside_backtest"
                    ),
                }
            )

    summary = {"symbol": symbol, "rows": len(df), **counts}
    return summary, details, pd.DataFrame(candidate_debug)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "baseline.yaml",
    )
    parser.add_argument("--symbol", default=None)
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--debug", action="store_true")
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

    summaries, details, debug_frames = [], [], []

    for path in paths:
        if path.name == "download_failures.txt" or not path.exists():
            continue

        summary, rows, debug = diagnose(
            path.stem,
            load_data(path),
            detector,
            start,
            end,
        )
        summaries.append(summary)
        details.extend(rows)

        if args.debug and not debug.empty:
            debug.insert(0, "symbol", path.stem)
            debug_frames.append(debug)

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

    if args.debug and debug_frames:
        debug_report = pd.concat(debug_frames, ignore_index=True)
        print("\nVALID VCP CANDIDATE DETAILS")
        print(debug_report.to_string(index=False))
        debug_report.to_csv(out / "vcp_candidate_debug.csv", index=False)
        print(f"Saved: {out / 'vcp_candidate_debug.csv'}")


if __name__ == "__main__":
    main()
