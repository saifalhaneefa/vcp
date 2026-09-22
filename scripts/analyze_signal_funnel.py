"""Analyze the VCP signal funnel without changing strategy rules.

The report counts trading days that survive each stage of the detector:
trend -> structural VCP -> fresh setup -> pivot break -> transition ->
breakout volume -> extension -> complete signal.

Run:
    python scripts/analyze_signal_funnel.py
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


def analyze(symbol: str, raw: pd.DataFrame, detector: VCPDetector,
            start: str, end: str) -> dict:
    df = add_indicators(raw, detector.cfg.trend)
    df["VolumeMA"] = df["Volume"].rolling(
        detector.cfg.vcp.volume_ma_days,
        min_periods=detector.cfg.vcp.volume_ma_days,
    ).mean()

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    counts = {
        "trend_days": 0,
        "structural_vcp_days": 0,
        "fresh_setup_days": 0,
        "pivot_break_days": 0,
        "transition_days": 0,
        "volume_confirmed_days": 0,
        "extension_ok_days": 0,
        "complete_signal_days": 0,
    }

    examples = []

    for i in range(1, len(df) - 1):
        date = pd.Timestamp(df.index[i])
        if date < start_ts or date > end_ts:
            continue
        if not detector.trend_ok(df.iloc[i]):
            continue

        counts["trend_days"] += 1
        setup = detector._setup(df, i)
        sequences = detector.find_vcp_sequences(setup)
        if not sequences:
            continue

        counts["structural_vcp_days"] += 1

        # Use the latest structural sequence, then apply the same freshness
        # and pivot-distance gates that find_setup uses.
        seq = sequences[-1]
        final = seq[-1]
        final_age = len(setup) - 1 - final.low_idx
        pivot = final.recovery_high

        if final_age > detector.cfg.vcp.max_final_contraction_age_bars:
            continue
        if pivot is None or pivot <= final.low:
            continue

        pivot_distance = pivot / final.low - 1.0
        if pivot_distance > detector.cfg.vcp.max_pivot_distance_from_final_low:
            continue

        counts["fresh_setup_days"] += 1

        row = df.iloc[i]
        close = float(row["Close"])
        pivot_break = close > float(pivot)
        if not pivot_break:
            continue
        counts["pivot_break_days"] += 1

        lookback = detector.cfg.vcp.breakout_transition_lookback_days
        prior = df.iloc[max(0, i - lookback):i]["Close"]
        transition_ok = prior.empty or float(prior.max()) <= float(pivot)
        if not transition_ok:
            continue
        counts["transition_days"] += 1

        ratio = (
            float(row["Volume"] / row["VolumeMA"])
            if row["VolumeMA"] > 0 else np.nan
        )
        volume_ok = (
            np.isfinite(ratio)
            and ratio >= detector.cfg.vcp.breakout_volume_multiple
        )
        if not volume_ok:
            continue
        counts["volume_confirmed_days"] += 1

        extension = close / float(pivot) - 1.0
        extension_ok = extension <= detector.cfg.vcp.max_breakout_extension
        if not extension_ok:
            continue
        counts["extension_ok_days"] += 1
        counts["complete_signal_days"] += 1

        if len(examples) < 10:
            examples.append(
                {
                    "date": date.date().isoformat(),
                    "pivot": float(pivot),
                    "breakout_pct": extension,
                    "breakout_volume_ratio": ratio,
                    "final_age_bars": final_age,
                    "pivot_distance": pivot_distance,
                }
            )

    result = {"symbol": symbol, "rows": len(df), **counts}
    return result, examples


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
    args = parser.parse_args()

    config = load_config(args.config)
    detector = VCPDetector(config)

    paths = sorted(
        p for p in args.data_dir.glob("*.csv")
        if p.name != "download_failures.txt"
    )

    summaries = []
    examples = []
    for path in paths:
        summary, rows = analyze(
            path.stem, load_data(path), detector, args.start, args.end
        )
        summaries.append(summary)
        for row in rows:
            examples.append({"symbol": path.stem, **row})

    if not summaries:
        raise SystemExit("No usable CSV files found.")

    report = pd.DataFrame(summaries)
    totals = report[
        [
            "trend_days",
            "structural_vcp_days",
            "fresh_setup_days",
            "pivot_break_days",
            "transition_days",
            "volume_confirmed_days",
            "extension_ok_days",
            "complete_signal_days",
        ]
    ].sum()

    total_row = {"symbol": "TOTAL", "rows": int(report["rows"].sum())}
    total_row.update(totals.to_dict())
    report = pd.concat([report, pd.DataFrame([total_row])], ignore_index=True)

    print("=" * 100)
    print("VCP SIGNAL FUNNEL")
    print("=" * 100)
    print(report.to_string(index=False))

    base = totals["trend_days"]
    if base > 0:
        print("\nStage retention from trend-qualified days:")
        for stage in [
            "structural_vcp_days",
            "fresh_setup_days",
            "pivot_break_days",
            "transition_days",
            "volume_confirmed_days",
            "extension_ok_days",
            "complete_signal_days",
        ]:
            print(f"  {stage:24s}: {totals[stage]:8.0f} / {base:.0f} "
                  f"= {totals[stage] / base:.2%}")

    out = PROJECT_ROOT / "reports"
    out.mkdir(exist_ok=True)
    report.to_csv(out / "signal_funnel.csv", index=False)
    pd.DataFrame(examples).to_csv(out / "signal_funnel_examples.csv", index=False)
    print(f"\nSaved: {out / 'signal_funnel.csv'}")
    print(f"Saved: {out / 'signal_funnel_examples.csv'}")


if __name__ == "__main__":
    main()
