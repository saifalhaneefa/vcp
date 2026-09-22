"""Sensitivity analysis for the 52-week-high proximity trend filter.

Does not change the baseline configuration. For each proximity threshold,
counts structurally valid/fresh/breakout/complete VCP candidates.

Run:
    python scripts/sweep_52w_high.py
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
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


def count_stages(data, cfg, start, end):
    detector = VCPDetector(cfg)
    totals = {
        "trend_days": 0,
        "structural_vcp_days": 0,
        "fresh_setup_days": 0,
        "pivot_break_days": 0,
        "transition_days": 0,
        "volume_confirmed_days": 0,
        "complete_signal_days": 0,
    }

    for raw in data.values():
        df = add_indicators(raw, cfg.trend)
        df["VolumeMA"] = df["Volume"].rolling(
            cfg.vcp.volume_ma_days,
            min_periods=cfg.vcp.volume_ma_days,
        ).mean()

        for i in range(1, len(df) - 1):
            date = pd.Timestamp(df.index[i])
            if date < start or date > end:
                continue
            if not detector.trend_ok(df.iloc[i]):
                continue

            totals["trend_days"] += 1
            setup = detector._setup(df, i)
            sequences = detector.find_vcp_sequences(setup)
            if not sequences:
                continue
            totals["structural_vcp_days"] += 1

            chosen = None
            for seq in reversed(sequences):
                final = seq[-1]
                age = len(setup) - 1 - final.low_idx
                pivot = final.recovery_high
                if age > cfg.vcp.max_final_contraction_age_bars:
                    continue
                if pivot is None or pivot <= final.low:
                    continue
                distance = pivot / final.low - 1.0
                if distance > cfg.vcp.max_pivot_distance_from_final_low:
                    continue
                chosen = (seq, float(pivot), age)
                break

            if chosen is None:
                continue
            totals["fresh_setup_days"] += 1

            seq, pivot, _ = chosen
            close = float(df.iloc[i]["Close"])
            if close <= pivot:
                continue
            totals["pivot_break_days"] += 1

            final = seq[-1]
            if final.recovery_high_idx is None:
                continue
            pivot_abs_idx = setup.index[final.recovery_high_idx]
            pivot_pos = df.index.get_loc(pivot_abs_idx)
            prior_since_pivot = df.iloc[pivot_pos + 1:i]["Close"]
            if not prior_since_pivot.empty and float(prior_since_pivot.max()) > pivot:
                continue
            totals["transition_days"] += 1

            ratio = (
                float(df.iloc[i]["Volume"] / df.iloc[i]["VolumeMA"])
                if df.iloc[i]["VolumeMA"] > 0 else np.nan
            )
            if not np.isfinite(ratio) or ratio < cfg.vcp.breakout_volume_multiple:
                continue
            totals["volume_confirmed_days"] += 1

            if close / pivot - 1.0 > cfg.vcp.max_breakout_extension:
                continue
            totals["complete_signal_days"] += 1

    return totals


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
        default="0.15,0.20,0.25,0.30,1.0",
        help="Maximum distance from 52W high. Use 1.0 for effectively no filter.",
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

    rows = []
    for threshold in [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]:
        cfg = replace(
            base,
            trend=replace(base.trend, max_distance_from_52w_high=threshold),
        )
        stages = count_stages(data, cfg, start, end)
        rows.append({
            "max_distance_from_52w_high": threshold,
            **stages,
        })

    report = pd.DataFrame(rows)
    print("=" * 100)
    print("52-WEEK-HIGH PROXIMITY SENSITIVITY")
    print("=" * 100)
    print(f"Period: {args.start} -> {args.end}")
    print(f"Symbols: {len(data)}")
    print(report.to_string(index=False))

    out = PROJECT_ROOT / "reports"
    out.mkdir(exist_ok=True)
    path = out / "52w_high_sensitivity.csv"
    report.to_csv(path, index=False)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
