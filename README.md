# VCP Backtester

Rules-based VCP backtesting project for NSE equities.

## Historical data

The repository includes a yfinance downloader:

```bash
python scripts/download_data.py
```

By default it:

- fetches the **current NIFTY 500 constituent list from NSE**
- downloads daily OHLCV history through yfinance
- saves one CSV per stock under `data/`
- downloads from **2013-01-01 to 2026-01-01** so the 2015-2025 backtest has indicator warm-up history

For a small test first:

```bash
python scripts/download_data.py --limit 20
```

For a few specific stocks:

```bash
python scripts/download_data.py --symbols RELIANCE TCS INFY HDFCBANK
```

The CSV format expected by the backtester is:

```text
Date,Open,High,Low,Close,Volume
```

## Run the current V1 runner

After data has been downloaded:

```bash
python scripts/backtest_portfolio.py --start 2015-01-01 --end 2025-12-31
```

**Important:** the current V1 runner tests each stock independently. It is not yet the final shared-cash, multi-position portfolio engine. V2 is being built for that purpose.

## Backtest methodology

Primary research period:

```text
2015-01-01 -> 2025-12-31
```

The extra 2013-2014 data is warm-up data for the 150/200-day moving averages and 252-day lookback calculations.

For the first research run, the universe is the current NIFTY 500. This introduces **survivorship bias** because today's constituents are not the same as the constituents that existed throughout 2015-2025. Results from this universe must therefore be treated as an initial research test, not a clean historical performance estimate.

A later version should use point-in-time historical constituents and delisted stocks where data is available.
