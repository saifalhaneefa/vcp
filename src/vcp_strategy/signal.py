from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Contraction:
    high_idx: int
    low_idx: int
    high: float
    low: float
    depth: float
    avg_volume: float


@dataclass(frozen=True)
class VCPSignal:
    signal_date: pd.Timestamp
    pivot: float
    breakout_volume_ratio: float
    stop_reference: float
    contractions: tuple


class VCPDetector:
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

    def _pivot_before(self, df, i):
        start = max(0, i - self.cfg.vcp.max_contraction_lookback_days)
        pstart = max(start, i - self.cfg.vcp.pivot_lookback_days)
        if pstart >= i:
            return np.nan
        return float(df.iloc[pstart:i].High.max())

    def find_signal(self, df, i):
        if i <= 0 or i >= len(df) - 1 or not self.trend_ok(df.iloc[i]):
            return None

        start = max(0, i - self.cfg.vcp.max_contraction_lookback_days)
        setup = df.iloc[start:i]
        cs = self.find_contractions(setup)
        if not self.valid(cs):
            return None

        pivot = self._pivot_before(df, i)
        row = df.iloc[i]
        ratio = float(row.Volume / row.VolumeMA) if row.VolumeMA > 0 else np.nan

        if not np.isfinite(ratio) or ratio < self.cfg.vcp.breakout_volume_multiple:
            return None
        if row.Close <= pivot:
            return None

        # A breakout is a transition from at/below the current resistance
        # to above it. Compare the previous close with the SAME pivot used
        # for today's breakout, rather than recomputing yesterday's pivot.
        previous_close = float(df.iloc[i - 1].Close)
        if previous_close >= pivot:
            return None

        return VCPSignal(
            df.index[i],
            pivot,
            ratio,
            float(row.Close) * (1 - self.cfg.risk.max_stop_loss_pct),
            tuple(cs),
        )

    def find_contractions(self, df):
        h = df.High.to_numpy(float)
        l = df.Low.to_numpy(float)
        v = df.Volume.to_numpy(float)
        w = 3
        peaks = [
            i for i in range(w, len(df) - w)
            if h[i] >= h[i - w:i + w + 1].max()
        ]
        troughs = [
            i for i in range(w, len(df) - w)
            if l[i] <= l[i - w:i + w + 1].min()
        ]
        out = []
        last = -10**9
        for p in peaks:
            after = [t for t in troughs if t > p]
            if not after:
                continue
            t = after[0]
            if t - last < self.cfg.vcp.min_contraction_separation:
                continue
            hi, lo = float(h[p]), float(l[t])
            if lo >= hi:
                continue
            depth = (hi - lo) / hi
            if depth < self.cfg.vcp.min_final_contraction:
                continue
            out.append(
                Contraction(
                    p,
                    t,
                    hi,
                    lo,
                    depth,
                    float(v[p:t + 1].mean()),
                )
            )
            last = t
        return out[-self.cfg.vcp.max_contractions:]

    def valid(self, cs):
        if len(cs) < self.cfg.vcp.min_contractions:
            return False
        d = [c.depth for c in cs]
        if (
            d[0] < self.cfg.vcp.min_first_contraction
            or not self.cfg.vcp.min_final_contraction <= d[-1] <= self.cfg.vcp.max_final_contraction
        ):
            return False
        if any(b > a * self.cfg.vcp.max_contraction_ratio for a, b in zip(d, d[1:])):
            return False
        v = [c.avg_volume for c in cs]
        return not any(b > a * self.cfg.vcp.max_volume_step for a, b in zip(v, v[1:]))
