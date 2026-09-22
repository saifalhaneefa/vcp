import numpy as np
import pandas as pd

from vcp_strategy import Backtester, PortfolioBacktester, VCPConfig, VCPDetector


def make_data(n=1000):
    idx = pd.date_range("2013-01-01", periods=n, freq="B")
    close = pd.Series(np.linspace(100, 220, n), index=idx)
    return pd.DataFrame(
        {
            "Open": close * 0.998,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": 200000,
        },
        index=idx,
    )


def test_smoke():
    df = make_data()
    assert isinstance(Backtester(VCPConfig()).run_symbol("TEST", df), list)


def test_portfolio_smoke():
    df = make_data()
    config = VCPConfig()
    result = PortfolioBacktester(config, starting_capital=1_000_000).run(
        {"TEST": df},
        start="2015-01-01",
        end="2015-12-31",
    )
    assert result.equity_curve.index.min() >= pd.Timestamp("2015-01-01")
    assert result.equity_curve.index.max() <= pd.Timestamp("2015-12-31")
    assert float(result.equity_curve.iloc[-1]) == 1_000_000.0
    assert result.trades == []


def test_sequential_contractions_are_progressively_shallower():
    cfg = VCPConfig()
    detector = VCPDetector(cfg)

    pivots = [
        (3, 100.0, 1200.0),
        (9, 85.0, 1050.0),
        (15, 98.0, 900.0),
        (21, 90.0, 780.0),
        (27, 97.0, 650.0),
        (33, 93.0, 560.0),
        (39, 97.5, 500.0),
    ]

    n = 45
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    close = np.zeros(n)
    volume = np.zeros(n)

    for (p0, v0, q0), (p1, v1, q1) in zip(pivots, pivots[1:]):
        for j in range(p0, p1 + 1):
            frac = (j - p0) / (p1 - p0)
            close[j] = v0 + frac * (v1 - v0)
            volume[j] = q0 + frac * (q1 - q0)

    # Flat warm-up before the first pivot.
    close[:3] = 95.0
    volume[:3] = 1200.0

    df = pd.DataFrame(
        {
            "Open": close,
            "High": close + 0.5,
            "Low": close - 0.5,
            "Close": close,
            "Volume": volume,
        },
        index=idx,
    )

    contractions = detector.find_contractions(df)

    assert len(contractions) == 3
    depths = [c.depth for c in contractions]
    assert depths[0] > depths[1] > depths[2]
    assert detector.valid(contractions)
