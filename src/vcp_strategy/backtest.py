from dataclasses import dataclass,asdict
import math
import pandas as pd
from .indicators import add_indicators
from .signal import VCPDetector

@dataclass
class Trade:
    symbol:str
    signal_date:object
    entry_date:object
    entry_price:float
    exit_date:object
    exit_price:float
    shares:int
    pnl:float
    return_pct:float
    stop_price:float

class Backtester:
    def __init__(self,config,starting_capital=1_000_000):
        self.cfg=config; self.starting_capital=float(starting_capital); self.detector=VCPDetector(config)
    def run_symbol(self,symbol,raw):
        df=add_indicators(raw,self.cfg.trend)
        df["VolumeMA"]=df.Volume.rolling(self.cfg.vcp.volume_ma_days,min_periods=self.cfg.vcp.volume_ma_days).mean()
        equity=self.starting_capital; trades=[]
        for i in range(1,len(df)-1):
            s=self.detector.find_signal(df,i)
            if not s: continue
            ei=i+1; entry=float(df.iloc[ei].Open); stop=s.stop_reference
            risk=equity*self.cfg.risk.portfolio_risk_per_trade
            shares=min(math.floor(risk/max(entry-stop,entry*.001)),math.floor(equity/entry))
            if shares<=0: continue
            xi,xp=self._exit(df,ei,stop)
            gross=(xp-entry)*shares
            cost=self._cost(entry*shares)+self._cost(xp*shares)
            pnl=gross-cost; equity+=pnl
            trades.append(Trade(symbol,s.signal_date,df.index[ei],entry,df.index[xi],xp,shares,pnl,pnl/(entry*shares),stop))
        return trades
    def _exit(self,df,i,stop):
        for j in range(i,len(df)):
            if df.iloc[j].Low<=stop: return j,min(float(df.iloc[j].Open),stop)
        return len(df)-1,float(df.iloc[-1].Close)
    def _cost(self,turnover): return turnover*(self.cfg.costs.commission_bps+self.cfg.costs.slippage_bps)/10000
    @staticmethod
    def metrics(trades,starting,ending,years):
        pnl=pd.Series([t.pnl for t in trades],dtype=float)
        if pnl.empty: return {"trades":0,"cagr":0.0,"max_drawdown":0.0}
        eq=starting+pnl.cumsum(); peak=eq.cummax()
        wins=pnl[pnl>0].sum(); losses=-pnl[pnl<0].sum()
        return {"trades":len(pnl),"cagr":float((ending/starting)**(1/years)-1),"win_rate":float((pnl>0).mean()),"profit_factor":float(wins/losses) if losses else float("inf"),"max_drawdown":float((eq/peak-1).min()),"ending_capital":float(ending)}
    @staticmethod
    def trades_frame(trades): return pd.DataFrame([asdict(t) for t in trades])
