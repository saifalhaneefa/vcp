import numpy as np
import pandas as pd

from vcp_strategy import Backtester, PortfolioBacktester, VCPConfig


def make_data(n=500):
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
