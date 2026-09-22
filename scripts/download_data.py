"""Download NSE equity history for the VCP backtest.

Default universe: current NIFTY 500 constituents from NSE.
Default download window: 2013-01-01 through 2026-01-01.

The extra history before 2015 supplies warm-up data for the 150/200-day
moving averages and 252-day high/low calculations.

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --limit 50
    python scripts/download_data.py --symbols RELIANCE TCS INFY
    python scripts/download_data.py --symbols-file symbols.txt
"""
from __future__ import annotations

import argparse
import io
import sys
import time
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = PROJECT_ROOT / "data"
NSE_NIFTY500_URL = "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv"


def fetch_nifty500_symbols() -> list[str]:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 Chrome/131 Safari/537.36"
        ),
        "Accept": "text/csv,*/*",
        "Referer": "https://www.nseindia.com/",
    }
    response = requests.get(NSE_NIFTY500_URL, headers=headers, timeout=30)
    response.raise_for_status()
    df = pd.read_csv(io.StringIO(response.text))
    if "Symbol" not in df.columns:
        raise ValueError(f"NSE CSV does not contain Symbol column: {df.columns.tolist()}")
    symbols = (
        df["Symbol"]
        .dropna()
        .astype(str)
        .str.strip()
        .drop_duplicates()
        .tolist()
    )
    return symbols


def read_symbols_file(path: Path) -> list[str]:
    symbols = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        symbols.append(line.split(",")[0].strip())
    return list(dict.fromkeys(symbols))


def normalize_symbols(symbols: list[str]) -> list[str]:
    result = []
    for symbol in symbols:
        if symbol.endswith(".NS") or symbol.endswith(".BO"):
            result.append(symbol)
        else:
            result.append(f"{symbol}.NS")
    return result


def save_symbol(df: pd.DataFrame, yahoo_symbol: str, out_dir: Path) -> bool:
    if df is None or df.empty:
        return False

    clean = df.copy()
    if isinstance(clean.columns, pd.MultiIndex):
        clean.columns = clean.columns.get_level_values(0)

    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = set(required) - set(clean.columns)
    if missing:
        return False

    clean = clean[required].copy()
    clean.index = pd.to_datetime(clean.index)
    clean.index.name = "Date"
    clean = clean.dropna()

    nse_symbol = yahoo_symbol.removesuffix(".NS").removesuffix(".BO")
    clean.to_csv(out_dir / f"{nse_symbol}.csv")
    return True


def download(
    symbols: list[str],
    out_dir: Path,
    start: str,
    end: str,
    chunk_size: int,
    pause_seconds: float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    yahoo_symbols = normalize_symbols(symbols)

    success = 0
    failed: list[str] = []

    for offset in range(0, len(yahoo_symbols), chunk_size):
        chunk = yahoo_symbols[offset : offset + chunk_size]
        print(
            f"Downloading {offset + 1}-{offset + len(chunk)} "
            f"of {len(yahoo_symbols)}..."
        )

        try:
            data = yf.download(
                tickers=chunk,
                start=start,
                end=end,
                interval="1d",
                auto_adjust=True,
                actions=False,
                threads=True,
                group_by="ticker",
                progress=True,
                timeout=30,
                # Do not use repair=True here. yfinance's repair feature
                # optionally imports SciPy; it is not required for our
                # standard daily OHLCV research dataset.
            )
        except Exception as exc:
            print(f"Chunk download failed: {exc}")
            failed.extend(chunk)
            time.sleep(pause_seconds)
            continue

        for yahoo_symbol in chunk:
            try:
                if len(chunk) == 1:
                    one = data
                else:
                    if not isinstance(data.columns, pd.MultiIndex):
                        one = pd.DataFrame()
                    elif yahoo_symbol in data.columns.get_level_values(0):
                        one = data[yahoo_symbol]
                    else:
                        one = pd.DataFrame()

                if save_symbol(one, yahoo_symbol, out_dir):
                    success += 1
                else:
                    failed.append(yahoo_symbol)
            except Exception as exc:
                print(f"  {yahoo_symbol}: failed to save ({exc})")
                failed.append(yahoo_symbol)

        time.sleep(pause_seconds)

    print("")
    print(f"Downloaded successfully: {success}")
    print(f"Failed/empty: {len(failed)}")

    if failed:
        failed_path = out_dir / "download_failures.txt"
        failed_path.write_text("\n".join(failed), encoding="utf-8")
        print(f"Failure list: {failed_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--start", default="2013-01-01")
    parser.add_argument("--end", default="2026-01-01")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--pause", type=float, default=2.0)
    parser.add_argument("--symbols", nargs="*", default=None)
    parser.add_argument("--symbols-file", type=Path, default=None)
    args = parser.parse_args()

    if args.symbols_file:
        symbols = read_symbols_file(args.symbols_file)
    elif args.symbols:
        symbols = args.symbols
    else:
        print("Fetching current NIFTY 500 constituents from NSE...")
        symbols = fetch_nifty500_symbols()

    if args.limit is not None:
        symbols = symbols[: args.limit]

    if not symbols:
        raise SystemExit("No symbols supplied.")

    print(f"Universe size: {len(symbols)}")
    print(f"History: {args.start} -> {args.end}")
    print(f"Output: {args.out_dir}")

    download(
        symbols=symbols,
        out_dir=args.out_dir,
        start=args.start,
        end=args.end,
        chunk_size=max(1, args.chunk_size),
        pause_seconds=max(0.0, args.pause),
    )


if __name__ == "__main__":
    main()
