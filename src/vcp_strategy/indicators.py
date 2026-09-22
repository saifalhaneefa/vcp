import pandas as pd

def add_indicators(df,cfg):
    required={"Open","High","Low","Close","Volume"}
    missing=required-set(df.columns)
    if missing: raise ValueError(f"Missing columns: {sorted(missing)}")
    x=df.copy().sort_index()
    for p in (cfg.sma_fast,cfg.sma_mid,cfg.sma_slow):
        x[f"SMA{p}"]=x.Close.rolling(p,min_periods=p).mean()
    x["Low52W"]=x.Low.rolling(cfg.low_52w_days,min_periods=cfg.low_52w_days).min()
    x["High52W"]=x.High.rolling(cfg.low_52w_days,min_periods=cfg.low_52w_days).max()
    return x
