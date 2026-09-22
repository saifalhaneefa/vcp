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
    """Mechanical approximation of a volatility-contraction pattern."""

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
        """Find confirmed local swings using a centered window."""
        w = self.cfg.vcp.swing_window
        if len(df) < 2 * w + 1:
            return []

        highs = df["High"].to_numpy(float)
        lows = df["Low"].to_numpy(float)
        points: list[SwingPoint] = []

        for j in range(w, len(df) - w):
            is_high = highs[j] >= np.max(highs[j - w:j + w + 1])
            is_low = lows[j] <= np.min(lows[j - w:j + w + 1])

            if is_high and is_low:
                # Prefer the side with the larger local excursion.
                left_high = highs[j - w:j].max()
                left_low = lows[j - w:j].min()
                right_high = highs[j + 1:j + w + 1].max()
                right_low = lows[j + 1:j + w + 1].min()
                high_excursion = highs[j] - max(left_high, right_high)
                low_excursion = min(left_low, right_low) - lows[j]
                kind = "HIGH" if high_excursion >= low_excursion else "LOW"
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

            prev = points[-1]
            if candidate.kind != prev.kind:
                points.append(candidate)
            elif candidate.kind == "HIGH" and candidate.price >= prev.price:
                points[-1] = candidate
            elif candidate.kind == "LOW" and candidate.price <= prev.price:
                points[-1] = candidate

        return points

    def find_contractions(self, df) -> list[Contraction]:
        """Convert consecutive HIGH -> LOW -> HIGH swings into contractions."""
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

        return contractions

    def _sequence_valid(self, seq: list[Contraction]) -> bool:
        if len(seq) < self.cfg.vcp.min_contractions:
            return False

        depths = [c.depth for c in seq]
        if depths[0] < self.cfg.vcp.min_first_contraction:
            return False
        if not self.cfg.vcp.min_final_contraction <= depths[-1] <= self.cfg.vcp.max_final_contraction:
            return False

        if any(
            later > earlier * self.cfg.vcp.max_contraction_ratio
            for earlier, later in zip(depths, depths[1:])
        ):
            return False

        # Volume is a VCP quality characteristic, not a structural requirement
        # at this stage. The hard volume confirmation is applied on the breakout
        # day (Volume / 50-day average >= breakout_volume_multiple). Keeping
        # setup volume soft avoids rejecting structurally valid patterns because
        # one contraction contains an unusual-volume event.

        return True

    def find_vcp_sequences(self, df) -> list[list[Contraction]]:
        """Return all contiguous valid contraction sequences."""
        contractions = self.find_contractions(df)
        sequences: list[list[Contraction]] = []
        n = len(contractions)
        min_n = self.cfg.vcp.min_contractions
        max_n = min(self.cfg.vcp.max_contractions, n)

        for length in range(min_n, max_n + 1):
            for start in range(0, n - length + 1):
                seq = contractions[start:start + length]
                if self._sequence_valid(seq):
                    sequences.append(seq)

        sequences.sort(
            key=lambda seq: (seq[-1].low_idx, len(seq), -seq[-1].depth)
        )
        return sequences

    def find_setup(self, df, i):
        if i <= 0 or i >= len(df):
            return None
        if not self.trend_ok(df.iloc[i]):
            return None

        setup = self._setup(df, i)
        sequences = self.find_vcp_sequences(setup)
        if not sequences:
            return None

        for contractions in reversed(sequences):
            final = contractions[-1]
            final_age = len(setup) - 1 - final.low_idx
            if final_age > self.cfg.vcp.max_final_contraction_age_bars:
                continue

            pivot = final.recovery_high
            if pivot is None or pivot <= final.low:
                continue

            pivot_distance = pivot / final.low - 1.0
            if pivot_distance > self.cfg.vcp.max_pivot_distance_from_final_low:
                continue

            return contractions, float(pivot), final_age, pivot_distance

        return None

    def find_signal(self, df, i):
        setup_result = self.find_setup(df, i)
        if setup_result is None:
            return None

        contractions, pivot, _, _ = setup_result
        row = df.iloc[i]
        ratio = float(row.Volume / row.VolumeMA) if row.VolumeMA > 0 else np.nan

        if not np.isfinite(ratio) or ratio < self.cfg.vcp.breakout_volume_multiple:
            return None

        close = float(row.Close)
        if close <= pivot:
            return None

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
