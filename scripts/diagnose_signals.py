"""Diagnose VCP candidates and show exactly where the structure is rejected."""
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


def sequence_reason(seq, cfg):
    reasons = []
    depths = [c.depth for c in seq]
    if depths[0] < cfg.min_first_contraction:
        reasons.append("first_depth")
    if not cfg.min_final_contraction <= depths[-1] <= cfg.max_final_contraction:
        reasons.append("final_depth")
    if any(
        later > earlier * cfg.max_contraction_ratio
        for earlier, later in zip(depths, depths[1:])
    ):
        reasons.append("depth_not_shrinking")
    volumes = [c.avg_volume for c in seq]
    if volumes[-1] > volumes[0] * 0.90:
        reasons.append("insufficient_volume_dryup")
    if any(
        later > earlier * cfg.max_volume_step
        for earlier, later in zip(volumes, volumes[1:])
    ):
        reasons.append("volume_step")
    return reasons


def diagnose(symbol, raw, detector):
    df = add_indicators(raw, detector.cfg.trend)
    df["VolumeMA"] = df["Volume"].rolling(
        detector.cfg.vcp.volume_ma_days,
        min_periods=detector.cfg.vcp.volume_ma_days,
    ).mean()

    counts = {
        "trend_ok": 0,
        "raw_contraction_count": 0,
        "three_wave_windows": 0,
        "depth_ok_windows": 0,
        "volume_quality_windows": 0,
        "valid_vcp_pattern": 0,
        "fresh_setup": 0,
        "volume_pass": 0,
        "pivot_pass": 0,
        "transition_pass": 0,
        "extension_ok": 0,
        "complete_signals": 0,
    }
    details = []
    best_debug = []

    for i in range(1, len(df) - 1):
        if not detector.trend_ok(df.iloc[i]):
            continue
        counts["trend_ok"] += 1

        setup = detector._setup(df, i)
        contractions = detector.find_contractions(setup)
        counts["raw_contraction_count"] = max(
            counts["raw_contraction_count"], len(contractions)
        )
        if len(contractions) < detector.cfg.vcp.min_contractions:
            continue
        counts["three_wave_windows"] += 1

        local_windows = []
        for start in range(
            0,
            len(contractions) - detector.cfg.vcp.min_contractions + 1,
        ):
            seq = contractions[start:start + detector.cfg.vcp.min_contractions]
            depths = [c.depth for c in seq]
            depth_ok = (
                depths[0] >= detector.cfg.vcp.min_first_contraction
                and detector.cfg.vcp.min_final_contraction
                <= depths[-1]
                <= detector.cfg.vcp.max_final_contraction
                and not any(
                    later > earlier * detector.cfg.vcp.max_contraction_ratio
                    for earlier, later in zip(depths, depths[1:])
                )
            )
            volumes = [c.avg_volume for c in seq]
            volume_ratio = (
                volumes[-1] / volumes[0]
                if volumes[0] > 0
                else np.nan
            )
            volume_drying = np.isfinite(volume_ratio) and volume_ratio <= 1.0

            if depth_ok:
                counts["depth_ok_windows"] += 1
            if volume_drying:
                counts["volume_quality_windows"] += 1

            local_windows.append({
                "start": start,
                "depths": depths,
                "volumes": volumes,
                "final_vs_first_volume": volume_ratio,
                "depth_ok": depth_ok,
            })

        setup_result = detector.find_setup(df, i)
        if setup_result is None:
            best_debug.append(
                {
                    "date": pd.Timestamp(df.index[i]).date().isoformat(),
                    "contractions": contractions,
                    "windows": local_windows,
                }
            )
            continue

        contractions, pivot, final_age, pivot_distance = setup_result
        counts["valid_vcp_pattern"] += 1
        if final_age <= detector.cfg.vcp.max_final_contraction_age_bars:
            counts["fresh_setup"] += 1

        row = df.iloc[i]
        ratio = (
            float(row.Volume / row.VolumeMA)
            if row.VolumeMA > 0 else float("nan")
        )
        close = float(row.Close)
        volume_ok = (
            np.isfinite(ratio)
            and ratio >= detector.cfg.vcp.breakout_volume_multiple
        )
        pivot_ok = close > pivot
        lookback = detector.cfg.vcp.breakout_transition_lookback_days
        prior = df.iloc[max(0, i - lookback):i]["Close"]
        transition_ok = prior.empty or float(prior.max()) <= pivot
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

        if volume_ok and pivot_ok and transition_ok and extension_ok:
            counts["complete_signals"] += 1
            next_date = pd.Timestamp(df.index[i + 1])
            raw_open = float(df.iloc[i + 1].Open)
            entry = raw_open * (
                1 + detector.cfg.costs.slippage_bps / 10000.0
            )
            stop = entry * (1 - detector.cfg.risk.max_stop_loss_pct)
            details.append({
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
            })

    return {"symbol": symbol, "rows": len(df), **counts}, details, best_debug


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

        summary, rows, failed = diagnose(path.stem, load_data(path), detector)
        summaries.append(summary)
        details.extend(rows)

        if args.debug and failed:
            print("\n" + "=" * 100)
            print(f"STRUCTURAL DIAGNOSTIC: {path.stem}")
            print("=" * 100)
            for item in failed[-5:]:
                print(f"\nCandidate date: {item['date']}")
                print("Raw sequential contractions:")
                for n, c in enumerate(item["contractions"], 1):
                    print(
                        f"  C{n}: high={c.high:.2f}, low={c.low:.2f}, "
                        f"depth={c.depth:.2%}, avg_volume={c.avg_volume:.0f}, "
                        f"recovery={c.recovery_high:.2f}"
                    )
                print("3-wave windows:")
                for w in item["windows"]:
                    print(
                        f"  window {w['start']}: "
                        f"depths={[round(x, 4) for x in w['depths']]}, "
                        f"volume={[round(x) for x in w['volumes']]}, "
                        f"final/first_volume={w['final_vs_first_volume']:.2f}x, "
                        f"depth_ok={w['depth_ok']}"
                    )

    summary_report = pd.DataFrame(summaries)
    detail_report = pd.DataFrame(details)

    if summary_report.empty:
        raise SystemExit("No usable CSV files found.")

    print("\nSIGNAL SUMMARY")
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
