"""Audit transition-qualified VCP candidates and their forward returns.

This is an observational research report. It does not change the strategy or
perform portfolio sizing. Candidates pass the structural, freshness, pivot,
and 1-day transition gates; volume is reported separately so we can inspect
the volume threshold without hiding candidates.

Run:
    python scripts/audit_vcp_candidates.py
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
    return (
        df.sort_values("Date")
        .drop_duplicates("Date")
        .set_index("Date")
    )


def median_forward_return(df: pd.DataFrame, i: int, days: int) -> float:
    if i + days >= len(df):
        return np.nan
    entry = float(df.iloc[i + 1]["Open"])
    exit_price = float(df.iloc[i + days]["Close"])
    if entry <= 0:
        return np.nan
    return exit_price / entry - 1.0


def audit_symbol(symbol: str, raw: pd.DataFrame, detector: VCPDetector,
                 start: pd.Timestamp, end: pd.Timestamp) -> list[dict]:
    df = add_indicators(raw, detector.cfg.trend)
    df["VolumeMA"] = df["Volume"].rolling(
        detector.cfg.vcp.volume_ma_days,
        min_periods=detector.cfg.vcp.volume_ma_days,
    ).mean()

    rows = []
    seen_dates = set()

    for i in range(1, len(df) - 1):
        date = pd.Timestamp(df.index[i])
        if date < start or date > end:
            continue
        if not detector.trend_ok(df.iloc[i]):
            continue

        setup = detector._setup(df, i)
        sequences = detector.find_vcp_sequences(setup)
        if not sequences:
            continue

        chosen = None
        for seq in reversed(sequences):
            final = seq[-1]
            age = len(setup) - 1 - final.low_idx
            pivot = final.recovery_high
            if age > detector.cfg.vcp.max_final_contraction_age_bars:
                continue
            if pivot is None or pivot <= final.low:
                continue
            pivot_distance = pivot / final.low - 1.0
            if pivot_distance > detector.cfg.vcp.max_pivot_distance_from_final_low:
                continue
            chosen = (seq, float(pivot), age, pivot_distance)
            break

        if chosen is None:
            continue

        seq, pivot, final_age, pivot_distance = chosen
        close = float(df.iloc[i]["Close"])
        if close <= pivot:
            continue

        lookback = detector.cfg.vcp.breakout_transition_lookback_days
        prior = df.iloc[max(0, i - lookback):i]["Close"]
        if not prior.empty and float(prior.max()) > pivot:
            continue

        ratio = (
            float(df.iloc[i]["Volume"] / df.iloc[i]["VolumeMA"])
            if df.iloc[i]["VolumeMA"] > 0 else np.nan
        )
        extension = close / pivot - 1.0
        volume_pass = np.isfinite(ratio) and ratio >= detector.cfg.vcp.breakout_volume_multiple

        # Avoid duplicate rows for repeated daily evaluation of the same
        # unchanged breakout setup.
        if date in seen_dates:
            continue
        seen_dates.add(date)

        depths = [c.depth for c in seq]
        volume_profile = [c.avg_volume for c in seq]
        rows.append({
            "symbol": symbol,
            "signal_date": date.date().isoformat(),
            "next_open": float(df.iloc[i + 1]["Open"]),
            "close": close,
            "pivot": pivot,
            "breakout_pct": extension,
            "volume_ratio": ratio,
            "volume_pass": bool(volume_pass),
            "final_depth_pct": depths[-1],
            "depths": ",".join(f"{x:.4f}" for x in depths),
            "final_age_bars": final_age,
            "pivot_distance_pct": pivot_distance,
            "final_first_volume_ratio": (
                volume_profile[-1] / volume_profile[0]
                if volume_profile[0] > 0 else np.nan
            ),
            "forward_5d": median_forward_return(df, i, 5),
            "forward_10d": median_forward_return(df, i, 10),
            "forward_20d": median_forward_return(df, i, 20),
            "forward_40d": median_forward_return(df, i, 40),
        })

    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "baseline.yaml")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2025-12-31")
    args = parser.parse_args()

    config = load_config(args.config)
    detector = VCPDetector(config)
    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)

    all_rows = []
    for path in sorted(args.data_dir.glob("*.csv")):
        if path.name == "download_failures.txt":
            continue
        all_rows.extend(
            audit_symbol(path.stem, load_data(path), detector, start, end)
        )

    report = pd.DataFrame(all_rows)
    if report.empty:
        raise SystemExit("No transition-qualified candidates found.")

    report = report.sort_values(
        ["signal_date", "symbol"], kind="stable"
    ).reset_index(drop=True)

    print("=" * 120)
    print("VCP CANDIDATE AUDIT")
    print("=" * 120)
    print(f"Candidates: {len(report)}")
    print(
        report[
            [
                "symbol", "signal_date", "close", "pivot", "breakout_pct",
                "volume_ratio", "volume_pass", "final_depth_pct",
                "final_age_bars", "pivot_distance_pct",
                "final_first_volume_ratio",
                "forward_5d", "forward_10d", "forward_20d", "forward_40d",
            ]
        ].to_string(index=False)
    )

    print("\nAGGREGATE BY VOLUME TEST")
    summary = report.groupby("volume_pass", dropna=False).agg(
        candidates=("symbol", "size"),
        median_5d=("forward_5d", "median"),
        median_10d=("forward_10d", "median"),
        median_20d=("forward_20d", "median"),
        median_40d=("forward_40d", "median"),
        mean_5d=("forward_5d", "mean"),
        mean_20d=("forward_20d", "mean"),
    )
    print(summary.to_string())

    out = PROJECT_ROOT / "reports"
    out.mkdir(exist_ok=True)
    path = out / "vcp_candidate_audit.csv"
    report.to_csv(path, index=False)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
