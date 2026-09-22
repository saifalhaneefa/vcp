# VCP Backtester

Rules-based VCP backtesting project for NSE equities.

## Historical data

The repository includes a yfinance downloader:

```bash
python scripts/download_data.py
```

By default it:

- fetches the current NIFTY 500 constituent list from NSE
- downloads daily OHLCV history through yfinance
- saves one CSV per stock under `data/`
- downloads from **2013-01-01 to 2026-01-01** so the 2015-2025 backtest has indicator warm-up history

For a small test first:

```bash
python scripts/download_data.py --limit 20
```

## Run the portfolio backtest

The current runner uses the multi-stock portfolio engine:

```bash
python scripts/backtest_portfolio.py
```

Default settings:

- Initial capital: **₹10,00,000**
- Backtest: **2015-01-01 -> 2025-12-31**
- Maximum simultaneous positions: **10**
- Portfolio risk per trade: **1%**
- Long-only, no leverage
- Signal generated on day t, entry at next trading day's open
- Stop checked using next day's open for gap-through-stop and intraday low otherwise
- Commission and slippage included from the baseline configuration

The runner writes:

```text
reports/portfolio_trades.csv
reports/portfolio_equity.csv
```

## Backtest methodology

The strategy rules are a mechanical approximation of the supplied VCP video transcript.

Primary research period:

```text
2015-01-01 -> 2025-12-31
```

The extra 2013-2014 data is warm-up data for the 150/200-day moving averages and 252-day lookback calculations.

For the first research run, the universe is the **current NIFTY 500**. This introduces survivorship bias because today's constituents are not the same as the constituents that existed throughout 2015-2025. Results from this universe are therefore an initial research test, not a clean historical performance estimate.

A later research version should use point-in-time historical constituents and include delisted securities where reliable data is available.

## Current limitations

The present VCP detector is deliberately mechanical and conservative enough for backtesting, but it does not yet fully model discretionary elements from the video, such as institutional accumulation/ distribution context, nuanced contraction identification, or a manually interpreted "tightness" structure.

Those rules should be refined only after we have a baseline result and inspect actual signal/trade examples.
