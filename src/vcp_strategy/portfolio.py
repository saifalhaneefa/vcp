from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np
import pandas as pd

from .indicators import add_indicators
from .signal import VCPDetector


@dataclass
class PortfolioPosition:
    symbol: str
    signal_date: pd.Timestamp
    entry_date: pd.Timestamp
    entry_price: float
    stop_price: float
    shares: int
    initial_risk: float


@dataclass
class PortfolioTrade:
    symbol: str
    signal_date: pd.Timestamp
    entry_date: pd.Timestamp
    entry_price: float
    exit_date: pd.Timestamp
    exit_price: float
    shares: int
    stop_price: float
    pnl: float
    return_pct: float
    r_multiple: float
    exit_reason: str


@dataclass
class PortfolioResult:
    trades: list[PortfolioTrade]
    equity_curve: pd.Series


class PortfolioBacktester:
    """Long-only, no-leverage, daily portfolio simulation.

    Signals are generated from data through day t and entered at the next
    trading day's open. The stop is constrained to the configured maximum
    loss from the actual entry price, so the 8% rule cannot become a larger
    loss merely because the stock gaps upward between signal and entry.
    """

    def __init__(self, config, starting_capital: float = 1_000_000.0):
        self.cfg = config
        self.starting_capital = float(starting_capital)
        self.detector = VCPDetector(config)

    def _prepare(self, raw: pd.DataFrame) -> pd.DataFrame:
        df = add_indicators(raw, self.cfg.trend)
        df["VolumeMA"] = df["Volume"].rolling(
            self.cfg.vcp.volume_ma_days,
            min_periods=self.cfg.vcp.volume_ma_days,
        ).mean()
        return df

    def _generate_signals(
        self, data: dict[str, pd.DataFrame]
    ) -> dict[pd.Timestamp, list[tuple[str, object]]]:
        signals: dict[pd.Timestamp, list[tuple[str, object]]] = {}
        for symbol, raw in data.items():
            df = self._prepare(raw)
            self._prepared[symbol] = df
            for i in range(1, len(df) - 1):
                signal = self.detector.find_signal(df, i)
                if signal is not None:
                    signals.setdefault(df.index[i], []).append((symbol, signal))
        return signals

    @staticmethod
    def _slipped_entry(price: float, bps: float) -> float:
        return price * (1.0 + bps / 10000.0)

    @staticmethod
    def _slipped_exit(price: float, bps: float) -> float:
        return price * (1.0 - bps / 10000.0)

    def run(
        self,
        data: dict[str, pd.DataFrame],
        start: str | None = None,
        end: str | None = None,
    ) -> PortfolioResult:
        if not data:
            raise ValueError("No symbol data supplied.")

        self._prepared: dict[str, pd.DataFrame] = {}
        signals = self._generate_signals(data)

        all_dates = sorted(
            set().union(*(df.index.tolist() for df in self._prepared.values()))
        )
        if start is not None:
            all_dates = [d for d in all_dates if d >= pd.Timestamp(start)]
        if end is not None:
            all_dates = [d for d in all_dates if d <= pd.Timestamp(end)]
        if not all_dates:
            raise ValueError("No trading dates remain after start/end filtering.")

        cash = self.starting_capital
        positions: dict[str, PortfolioPosition] = {}
        pending: dict[pd.Timestamp, list[tuple[str, object]]] = {}
        trades: list[PortfolioTrade] = []
        equity_values: list[float] = []
        equity_dates: list[pd.Timestamp] = []

        for date in all_dates:
            # 1. Exit existing positions at today's open/intraday low if stop is hit.
            for symbol, position in list(positions.items()):
                row = self._row_on_date(symbol, date)
                if row is None:
                    continue

                low = float(row["Low"])
                open_price = float(row["Open"])
                exit_price = None
                reason = None

                if open_price <= position.stop_price:
                    exit_price = open_price
                    reason = "stop_gap"
                elif low <= position.stop_price:
                    exit_price = position.stop_price
                    reason = "stop"

                if exit_price is not None:
                    fill = self._slipped_exit(
                        exit_price, self.cfg.costs.slippage_bps
                    )
                    turnover = fill * position.shares
                    exit_commission = self._commission(turnover)
                    cash += turnover - exit_commission

                    gross = (fill - position.entry_price) * position.shares
                    entry_commission = self._entry_commission(position)
                    pnl = gross - entry_commission - exit_commission

                    trades.append(
                        PortfolioTrade(
                            symbol,
                            position.signal_date,
                            position.entry_date,
                            position.entry_price,
                            date,
                            fill,
                            position.shares,
                            position.stop_price,
                            pnl,
                            pnl / (position.entry_price * position.shares),
                            pnl / position.initial_risk
                            if position.initial_risk > 0
                            else np.nan,
                            reason,
                        )
                    )
                    del positions[symbol]

            # 2. Execute yesterday's signals at today's open.
            entries = pending.pop(date, [])
            if entries:
                ranked = sorted(
                    entries,
                    key=lambda x: (
                        -float(x[1].breakout_volume_ratio),
                        float(x[1].contractions[-1].depth),
                        x[0],
                    ),
                )

                current_equity_for_sizing = self._mark_equity(
                    cash, positions, date
                )
                risk_budget = (
                    current_equity_for_sizing
                    * self.cfg.risk.portfolio_risk_per_trade
                )

                for symbol, signal in ranked:
                    if symbol in positions:
                        continue
                    if len(positions) >= self.cfg.risk.max_simultaneous_positions:
                        break

                    row = self._row_on_date(symbol, date)
                    if row is None:
                        continue

                    raw_open = float(row["Open"])
                    entry = self._slipped_entry(
                        raw_open, self.cfg.costs.slippage_bps
                    )

                    # The supplied VCP rules cap the stop at 8% below the
                    # actual buy price. A pattern-derived stop can be tighter,
                    # but never wider than the configured maximum.
                    max_loss_stop = entry * (
                        1.0 - self.cfg.risk.max_stop_loss_pct
                    )
                    pattern_stop = float(signal.stop_reference)
                    stop = max(pattern_stop, max_loss_stop)

                    per_share_risk = entry - stop
                    if per_share_risk <= 0:
                        continue

                    max_risk_shares = math.floor(risk_budget / per_share_risk)
                    available_cash = max(0.0, cash)
                    cost_rate = self.cfg.costs.commission_bps / 10000.0
                    max_cash_shares = math.floor(
                        available_cash / (entry * (1.0 + cost_rate))
                    )
                    shares = min(max_risk_shares, max_cash_shares)
                    if shares <= 0:
                        continue

                    turnover = entry * shares
                    entry_commission = self._commission(turnover)
                    total_debit = turnover + entry_commission
                    if total_debit > cash:
                        continue

                    cash -= total_debit
                    positions[symbol] = PortfolioPosition(
                        symbol=symbol,
                        signal_date=pd.Timestamp(signal.signal_date),
                        entry_date=pd.Timestamp(date),
                        entry_price=entry,
                        stop_price=stop,
                        shares=shares,
                        initial_risk=per_share_risk * shares,
                    )

            # 3. Queue today's signals for the next available trading date.
            next_date = self._next_available_date(all_dates, date)
            if next_date is not None:
                for symbol, signal in signals.get(date, []):
                    pending.setdefault(next_date, []).append((symbol, signal))

            # 4. Mark portfolio at today's close.
            equity_values.append(self._mark_equity(cash, positions, date))
            equity_dates.append(date)

        # 5. Liquidate remaining positions at the final close.
        last_date = all_dates[-1]
        for symbol, position in list(positions.items()):
            row = self._row_on_date(symbol, last_date)
            if row is None:
                continue

            raw_exit = float(row["Close"])
            fill = self._slipped_exit(raw_exit, self.cfg.costs.slippage_bps)
            turnover = fill * position.shares
            exit_commission = self._commission(turnover)
            cash += turnover - exit_commission

            gross = (fill - position.entry_price) * position.shares
            entry_commission = self._entry_commission(position)
            pnl = gross - entry_commission - exit_commission

            trades.append(
                PortfolioTrade(
                    symbol,
                    position.signal_date,
                    position.entry_date,
                    position.entry_price,
                    last_date,
                    fill,
                    position.shares,
                    position.stop_price,
                    pnl,
                    pnl / (position.entry_price * position.shares),
                    pnl / position.initial_risk
                    if position.initial_risk > 0
                    else np.nan,
                    "end_of_test",
                )
            )
            del positions[symbol]

        if equity_dates:
            equity_values[-1] = cash

        return PortfolioResult(
            trades=trades,
            equity_curve=pd.Series(
                equity_values, index=equity_dates, name="equity"
            ),
        )

    def _row_on_date(self, symbol: str, date: pd.Timestamp):
        df = self._prepared[symbol]
        if date not in df.index:
            return None
        return df.loc[date]

    @staticmethod
    def _next_available_date(
        dates: list[pd.Timestamp], date: pd.Timestamp
    ) -> pd.Timestamp | None:
        pos = dates.index(date)
        return dates[pos + 1] if pos + 1 < len(dates) else None

    def _mark_equity(
        self,
        cash: float,
        positions: dict[str, PortfolioPosition],
        date: pd.Timestamp,
    ) -> float:
        value = cash
        for symbol, position in positions.items():
            row = self._row_on_date(symbol, date)
            if row is not None:
                value += float(row["Close"]) * position.shares
        return value

    def _entry_commission(self, position: PortfolioPosition) -> float:
        return self._commission(position.entry_price * position.shares)

    def _commission(self, turnover: float) -> float:
        return turnover * self.cfg.costs.commission_bps / 10000.0

    @staticmethod
    def metrics(
        result: PortfolioResult, starting_capital: float = 1_000_000.0
    ) -> dict:
        start = float(starting_capital)
        end = float(result.equity_curve.iloc[-1])
        days = max(
            1.0,
            (result.equity_curve.index[-1] - result.equity_curve.index[0]).days,
        )
        years = days / 365.25

        daily_returns = result.equity_curve.pct_change().dropna()
        mean = float(daily_returns.mean()) if not daily_returns.empty else 0.0
        std = float(daily_returns.std(ddof=1)) if len(daily_returns) > 1 else 0.0
        sharpe = mean / std * np.sqrt(252.0) if std > 0 else 0.0

        peak = result.equity_curve.cummax()
        drawdown = result.equity_curve / peak - 1.0
        pnl = np.array([t.pnl for t in result.trades], dtype=float)
        wins = pnl[pnl > 0].sum() if len(pnl) else 0.0
        losses = -pnl[pnl < 0].sum() if len(pnl) else 0.0

        return {
            "trades": int(len(pnl)),
            "cagr": float((end / start) ** (1.0 / years) - 1.0)
            if start > 0 and end > 0
            else -1.0,
            "win_rate": float((pnl > 0).mean()) if len(pnl) else 0.0,
            "profit_factor": float(wins / losses) if losses > 0 else float("inf"),
            "max_drawdown": float(drawdown.min()),
            "sharpe": float(sharpe),
            "ending_capital": end,
            "net_profit": end - start,
            "average_trade_return": float(
                np.mean([t.return_pct for t in result.trades])
            )
            if result.trades
            else 0.0,
            "average_r_multiple": float(
                np.nanmean([t.r_multiple for t in result.trades])
            )
            if result.trades
            else 0.0,
        }

    @staticmethod
    def trades_frame(result: PortfolioResult) -> pd.DataFrame:
        return pd.DataFrame([asdict(t) for t in result.trades])
