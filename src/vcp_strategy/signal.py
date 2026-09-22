from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SwingPoint:
    index: int
    kind: str
    price: float


@dataclass(frozen=True)
class Contraction:
    high_idx: int
    low_idx: int
    high: float
    low: float
    depth: float
    avg_volume: float
    recovery_high_idx: int | None = None
    recovery_high: float | None = None


@dataclass(frozen=True)
class VCPSignal:
    signal_date: pd.Timestamp
    pivot: float
    breakout_volume_ratio: float
    stop_reference: float
    contractions: tuple


class VCPDetector:
    """Mechanical approximation of a volatility-contraction pattern.

    The detector deliberately separates:
      1) trend template,
      2) sequential contraction structure,
      3) final pivot/resistance,
      4) breakout confirmation.

    Swing points use a centered confirmation window. For a candidate on day t,
    only swing points that were confirmable using data through t-1 are used.
    """

    def __init__(self, config):
        self.cfg = config

    def trend_ok(self, row):
        cols = ["Close", "SMA50", "SMA150", "SMA200", "Low52W", "High52W"]
        if any(pd.isna(row.get(c)) for c in cols):
            return False
        close = float(row.Close)
        return (
            close > row.SMA50 > row.SMA150 > row.SMA200
            and close >= row.Low52W * 1.30
            and close >= row.High52W * (1 - self.cfg.trend.max_distance_from_52w_high)
        )

    def _setup(self, df, i):
        start = max(0, i - self.cfg.vcp.max_contraction_lookback_days)
        return df.iloc[start:i]

    def find_swings(self, df) -> list[SwingPoint]:
        """Find confirmed alternating swing highs/lows and remove duplicates."""
        w = self.cfg.vcp.swing_window
        if len(df) < 2 * w + 1:
            return []

        highs = df["High"].to_numpy(float)
        lows = df["Low"].to_numpy(float)
        points: list[SwingPoint] = []

        for j in range(w, len(df) - w):
            is_high = highs[j] >= np.max(highs[j - w:j + w + 1])
            is_low = lows[j] <= np.min(lows[j - w:j + w + 1])

            # A wide candle could technically qualify as both. Keep the
            # stronger side so the sequence remains strictly alternating.
            if is_high and is_low:
                up_range = highs[j] - lows[j]
                left_range = highs[j - w:j].max() - lows[j - w:j].min()
                right_range = highs[j + 1:j + w + 1].max() - lows[j + 1:j + w + 1].min()
                kind = "HIGH" if up_range >= max(left_range, right_range) else "LOW"
                price = highs[j] if kind == "HIGH" else lows[j]
                candidate = SwingPoint(j, kind, float(price))
            elif is_high:
                candidate = SwingPoint(j, "HIGH", float(highs[j]))
            elif is_low:
                candidate = SwingPoint(j, "LOW", float(lows[j]))
            else:
                continue

            if not points:
                points.append(candidate)
                continue

            previous = points[-1]
            if candidate.kind != previous.kind:
                points.append(candidate)
                continue

            # Consecutive same-type pivots are collapsed into the more
            # extreme point.
            if candidate.kind == "HIGH" and candidate.price >= previous.price:
                points[-1] = candidate
            elif candidate.kind == "LOW" and candidate.price <= previous.price:
                points[-1] = candidate

        return points

    def find_contractions(self, df) -> list[Contraction]:
        """Build contractions only from sequential HIGH -> LOW -> HIGH swings."""
        swings = self.find_swings(df)
        contractions: list[Contraction] = []

        for k in range(len(swings) - 2):
            a, b, c = swings[k:k + 3]
            if not (a.kind == "HIGH" and b.kind == "LOW" and c.kind == "HIGH"):
                continue

            if b.index - a.index < self.cfg.vcp.min_contraction_separation:
                continue

            depth = (a.price - b.price) / a.price
            if depth < self.cfg.vcp.min_final_contraction:
                continue

            # The rebound should be real, not merely a one-bar bounce.
            rebound = (c.price - b.price) / b.price
            if rebound < self.cfg.vcp.min_rebound_from_low:
                continue

            avg_volume = float(df.iloc[a.index:b.index + 1]["Volume"].mean())

            contractions.append(
                Contraction(
                    high_idx=a.index,
                    low_idx=b.index,
                    high=a.price,
                    low=b.price,
                    depth=depth,
                    avg_volume=avg_volume,
                    recovery_high_idx=c.index,
                    recovery_high=c.price,
                )
            )

        return contractions[-self.cfg.vcp.max_contractions:]

    def valid(self, contractions: list[Contraction]) -> bool:
        if len(contractions) < self.cfg.vcp.min_contractions:
            return False

        depths = [c.depth for c in contractions]

        if depths[0] < self.cfg.vcp.min_first_contraction:
            return False

        if not (
            self.cfg.vcp.min_final_contraction
            <= depths[-1]
            <= self.cfg.vcp.max_final_contraction
        ):
            return False

        # Successive pullbacks must become progressively shallower.
        if any(
            later > earlier * self.cfg.vcp.max_contraction_ratio
            for earlier, later in zip(depths, depths[1:])
        ):
            return False

        volumes = [c.avg_volume for c in contractions]
        if any(
            later > earlier * self.cfg.vcp.max_volume_step
            for earlier, later in zip(volumes, volumes[1:])
        ):
            return False

        # Recovery highs should not collapse dramatically. This guards
        # against stitching together unrelated swings.
        for a, b in zip(contractions, contractions[1:]):
            if a.recovery_high is None or b.high is None:
                continue
            if b.high < a.recovery_high * 0.70:
                return False

        return True

    def _pivot_for_setup(self, setup, final: Contraction):
        swings = self.find_swings(setup)
        after_low = [
            s for s in swings
            if s.kind == "HIGH"
            and s.index > final.low_idx
        ]
        if not after_low:
            return np.nan

        # The latest confirmed recovery high is the working pivot. This
        # naturally stays close to the right edge of the VCP.
        pivot = float(after_low[-1].price)
        if pivot <= final.low:
            return np.nan
        return pivot

    def find_setup(self, df, i):
        if i <= 0 or i >= len(df):
            return None
        if not self.trend_ok(df.iloc[i]):
            return None

        setup = self._setup(df, i)
        contractions = self.find_contractions(setup)
        if not self.valid(contractions):
            return None

        final = contractions[-1]
        pivot = self._pivot_for_setup(setup, final)
        if not np.isfinite(pivot):
            return None

        final_age = len(setup) - 1 - final.low_idx
        if final_age > self.cfg.vcp.max_final_contraction_age_bars:
            return None

        pivot_distance = pivot / final.low - 1.0
        if pivot_distance > self.cfg.vcp.max_pivot_distance_from_final_low:
            return None

        return contractions, pivot, final_age, pivot_distance

    def find_signal(self, df, i):
        setup_result = self.find_setup(df, i)
        if setup_result is None:
            return None

        contractions, pivot, _, _ = setup_result
        row = df.iloc[i]

        ratio = float(row.Volume / row.VolumeMA) if row.VolumeMA > 0 else np.nan
        if (
            not np.isfinite(ratio)
            or ratio < self.cfg.vcp.breakout_volume_multiple
        ):
            return None

        close = float(row.Close)
        if close <= pivot:
            return None

        # Require a genuine transition through resistance rather than a
        # continuation day. Look back several closes to avoid duplicates.
        lookback = self.cfg.vcp.breakout_transition_lookback_days
        prior = df.iloc[max(0, i - lookback):i]["Close"]
        if not prior.empty and float(prior.max()) > pivot:
            return None

        extension = close / pivot - 1.0
        if extension > self.cfg.vcp.max_breakout_extension:
            return None

        return VCPSignal(
            df.index[i],
            pivot,
            ratio,
            close * (1 - self.cfg.risk.max_stop_loss_pct),
            tuple(contractions),
        )
