"""var_es: rolling-window VaR/ES backtests of four tail-risk models."""

from ._types import RiskEstimate
from .backtest import (
    christoffersen_cc,
    christoffersen_independence,
    kupiec_pof,
    rolling_backtest,
    summarize,
)
from .evt import GPDTail, evt_pot_var_es, fit_pot
from .garch import GarchT, fit_garch_t, garch_t_var_es
from .gbm import fit_gbm, gbm_mc_var_es
from .historical import historical_var_es

__all__ = [
    "GPDTail",
    "GarchT",
    "RiskEstimate",
    "christoffersen_cc",
    "christoffersen_independence",
    "evt_pot_var_es",
    "fit_garch_t",
    "fit_gbm",
    "fit_pot",
    "garch_t_var_es",
    "gbm_mc_var_es",
    "historical_var_es",
    "kupiec_pof",
    "rolling_backtest",
    "summarize",
]
__version__ = "0.1.0"
