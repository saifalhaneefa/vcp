"""Diagnose VCP candidates after the sequential swing-structure rebuild."""
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
    return (
        df.sort_values("Date")
        .drop_duplicates("Date")
        .set_index("Date")
    )


def diagnose(symbol: str, raw: pd.DataFrame, detector: VCPDetector):
    df = add_indicators(raw, detector.cfg.trend)
    df["VolumeMA"] = df["Volume"].rolling(
        detector.cfg.vcp.volume_ma_days,
        min_periods=detector.cfg.vcp.volume_ma_days,
    ).mean()

    counts = {
        "trend_ok": 0,
        "valid_vcp_pattern": 0,
        "fresh_setup": 0,
        "volume_pass": 0,
        "pivot_pass": 0,
        "transition_pass": 0,
        "extension_ok": 0,
        "complete_signals": 0,
    }
    details = []

    for i in range(1, len(df) - 1):
        if not detector.trend_ok(df.iloc[i]):
            continue
        counts["trend_ok"] += 1

        setup_result = detector.find_setup(df, i)
        if setup_result is None:
            continue

        contractions, pivot, final_age, pivot_distance = setup_result
        counts["valid_vcp_pattern"] += 1
        if final_age <= detector.cfg.vcp.max_final_contraction_age_bars:
            counts["fresh_setup"] += 1

        row = df.iloc[i]
        close = float(row.Close)
        ratio = (
            float(row.Volume / row.VolumeMA)
            if row.VolumeMA > 0
            else float("nan")
        )
        previous_close = float(df.iloc[i - 1].Close)

        volume_ok = (
            pd.notna(ratio)
            and ratio >= detector.cfg.vcp.breakout_volume_multiple
        )
        pivot_ok = close > pivot
        lookback = detector.cfg.vcp.breakout_transition_lookback_days
        prior = df.iloc[max(0, i - lookback):i]["Close"]
        transition_ok = prior.empty or float(prior.max()) <= pivot
        extension_ok = (
            close / pivot - 1.0
            <= detector.cfg.vcp.max_breakout_extension
        )

        if volume_ok:
            counts["volume_pass"] += 1
        if pivot_ok:
            counts["pivot_pass"] += 1
        if transition_ok:
            counts["transition_pass"] += 1
        if extension_ok:
            counts["extension_ok"] += 1

        complete = (
            volume_ok
            and pivot_ok
            and transition_ok
            and extension_ok
        )
        if complete:
            counts["complete_signals"] += 1
            signal = detector.find_signal(df, i)
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
                    "breakout_pct": close / pivot - 1.0,
                    "final_age_bars": final_age,
                    "pivot_distance_from_final_low": pivot_distance,
                    "breakout_volume_ratio": ratio,
                    "entry_open": raw_open,
                    "simulated_entry": entry,
                    "stop": stop,
                    "planned_loss_pct": 1 - stop / entry,
                }
            )

    return {"symbol": symbol, "rows": len(df), **counts}, pd.DataFrame(details)


def debug_symbol(symbol: str, raw: pd.DataFrame, detector: VCPDetector):
    df = add_indicators(raw, detector.cfg.trend)
    df["VolumeMA"] = df["Volume"].rolling(
        detector.cfg.vcp.volume_ma_days,
        min_periods=detector.cfg.vcp.volume_ma_days,
    ).mean()

    rows = []
    for i in range(1, len(df) - 1):
        if not detector.trend_ok(df.iloc[i]):
            continue

        setup_result = detector.find_setup(df, i)
        if setup_result is None:
            continue

        contractions, pivot, final_age, pivot_distance = setup_result
        row = df.iloc[i]
        ratio = (
            float(row.Volume / row.VolumeMA)
            if row.VolumeMA > 0
            else float("nan")
        )

        print("\n" + "=" * 90)
        print(f"VCP CANDIDATE: {symbol} | {df.index[i].date()}")
        print("=" * 90)
        print(f"Close: {float(row.Close):.2f}")
        print(f"Pivot: {pivot:.2f}")
        print(f"Previous close: {float(df.iloc[i - 1].Close):.2f}")
        print(f"Volume ratio: {ratio:.2f}x")
        print(f"Final contraction age: {final_age} bars")
        print(f"Pivot distance from final low: {pivot_distance:.2%}")
        print("Contractions:")

        setup = df.iloc[max(0, i - detector.cfg.vcp.max_contraction_lookback_days):i]
        for n, c in enumerate(contractions, 1):
            print(
                f"  C{n}: "
                f"{setup.index[c.high_idx].date()} {c.high:.2f} -> "
                f"{setup.index[c.low_idx].date()} {c.low:.2f} "
                f"depth={c.depth:.2%}; "
                f"recovery={setup.index[c.recovery_high_idx].date()} "
                f"{c.recovery_high:.2f}"
            )

        recent = df.iloc[max(0, i - 10):i + 1][
            ["Open", "High", "Low", "Close", "Volume"]
        ]
        print("\nRecent bars:")
        print(recent.to_string())

        rows.append(
            {
                "symbol": symbol,
                "candidate_date": pd.Timestamp(df.index[i]).date().isoformat(),
                "pivot": pivot,
                "final_age_bars": final_age,
                "pivot_distance_from_final_low": pivot_distance,
                "breakout_volume_ratio": ratio,
                "contractions": " | ".join(
                    f"C{n}:{setup.index[c.high_idx].date()}->{setup.index[c.low_idx].date()} "
                    f"{c.high:.2f}->{c.low:.2f} ({c.depth:.2%}) "
                    f"recovery={c.recovery_high:.2f}"
                    for n, c in enumerate(contractions, 1)
                ),
            }
        )

    return pd.DataFrame(rows)


def main():
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

    summaries = []
    all_details = []
    all_debug = []

    for path in paths:
        if path.name == "download_failures.txt" or not path.exists():
            continue

        raw = load_data(path)
        summary, detail = diagnose(path.stem, raw, detector)
        summaries.append(summary)

        if not detail.empty:
            detail = detail[
                (pd.to_datetime(detail["entry_date"]) >= start)
                & (pd.to_datetime(detail["entry_date"]) <= end)
            ]
            all_details.append(detail)

        if args.debug:
            debug = debug_symbol(path.stem, raw, detector)
            if not debug.empty:
                all_debug.append(debug)

    summary_report = pd.DataFrame(summaries)
    detail_report = (
        pd.concat(all_details, ignore_index=True)
        if all_details
        else pd.DataFrame()
    )

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

    if all_debug:
        debug_report = pd.concat(all_debug, ignore_index=True)
        debug_report.to_csv(out / "vcp_candidate_debug.csv", index=False)
        print(f"Saved: {out / 'vcp_candidate_debug.csv'}")


if __name__ == "__main__":
    main()
