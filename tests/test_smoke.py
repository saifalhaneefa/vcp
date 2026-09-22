import numpy as np,pandas as pd
from vcp_strategy import Backtester,VCPConfig

def test_smoke():
    n=500; idx=pd.date_range("2024-01-01",periods=n,freq="B"); c=pd.Series(np.linspace(100,220,n),index=idx)
    df=pd.DataFrame({"Open":c*.998,"High":c*1.01,"Low":c*.99,"Close":c,"Volume":200000},index=idx)
    assert isinstance(Backtester(VCPConfig()).run_symbol("TEST",df),list)
