from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass(frozen=True)
class TrendConfig:
    sma_fast: int = 50
    sma_mid: int = 150
    sma_slow: int = 200
    low_52w_days: int = 252
    max_distance_from_52w_high: float = 0.15


@dataclass(frozen=True)
class PatternConfig:
    min_contractions: int = 3
    max_contractions: int = 5
    min_first_contraction: float = 0.05
    min_final_contraction: float = 0.01
    max_final_contraction: float = 0.10
    max_contraction_ratio: float = 0.90
    min_contraction_separation: int = 5
    max_contraction_lookback_days: int = 150
    swing_window: int = 3
    min_rebound_from_low: float = 0.03
    max_final_contraction_age_bars: int = 20
    max_pivot_distance_from_final_low: float = 0.20
    max_breakout_extension: float = 0.10
    breakout_transition_lookback_days: int = 5
    pivot_lookback_days: int = 30
    volume_ma_days: int = 50
    max_volume_step: float = 1.10
    breakout_volume_multiple: float = 1.50


@dataclass(frozen=True)
class RiskConfig:
    max_stop_loss_pct: float = 0.08
    portfolio_risk_per_trade: float = 0.01
    max_simultaneous_positions: int = 10


@dataclass(frozen=True)
class CostConfig:
    commission_bps: float = 5
    slippage_bps: float = 10


@dataclass(frozen=True)
class VCPConfig:
    trend: TrendConfig = TrendConfig()
    vcp: PatternConfig = PatternConfig()
    risk: RiskConfig = RiskConfig()
    costs: CostConfig = CostConfig()


def load_config(path):
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return VCPConfig(
        TrendConfig(**raw.get("trend", {})),
        PatternConfig(**raw.get("vcp", {})),
        RiskConfig(**raw.get("risk", {})),
        CostConfig(**raw.get("costs", {})),
    )
