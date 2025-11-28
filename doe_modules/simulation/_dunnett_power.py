import numpy as np
import pandas as pd
import rpy2.robjects as ro
import rpy2.robjects.vectors as ro_vectors
from rpy2.robjects.packages import importr, isinstalled
from rpy2.robjects import numpy2ri

from ._abstract import AbstractSimulator
from ._dunnett import Dunnett
from ._anova_power import sigma2


numpy2ri.activate()

utils = importr('utils')
base = importr('base')
packages = ["mvtnorm", "stats"]
repos = "https://cloud.r-project.org/"
rscript = f"{os.path.dirname(__file__)}/dunnett_power.R"
func_name = "dunnett_power_analytic"
R_FUNC = None


for pkg in packages:
    if not isinstalled(pkg): 
        utils.install_packages(pkg, repos=repos)
    base.suppressPackageStartupMessages(
        base.library(pkg, character_only=ro_vectors.BoolVector([True]))
    )


with open(rscript) as f:
    ro.r(f.read())


def _initialize_r_func():
    global R_FUNC
    if R_FUNC is None:
        R_FUNC = ro.r[func_name]


def dunnett_power(
    simulation: AbstractSimulator,
    alpha: float = 0.05,
    power_type: str = "per_comparison_power",
) -> np.ndarray:
    assert power_type in ["per_comparison_power", "familywise_power"], \
        f"power_type should be either 'per_comparison_power' or 'familywise_power', got {power_type}"
    dunnett_model = Dunnett(simulation)
    if dunnett_model.n_rep > 1:
        dunnett_model.summary()
        _initialize_r_func()
        power_dict = (
            lambda listvector: dict(zip(listvector.names, listvector))
        )(
            R_FUNC(
                mu0=float(dunnett_model.baseline.y),
                mu_t=dunnett_model.coef.y.values.ravel(),
                n0=dunnett_model.n_rep,
                nt=dunnett_model.n_rep,
                sigma=np.sqrt(sigma2(dunnett_model.simulation).item()),
                alpha=alpha
            )
        )

    else:
        power_dict =  {
            "df": np.nan,
            "crit": np.nan,
            "per_comparison_power": [np.nan] * (len(dunnett_model.group) - 1),
            "familywise_power": np.nan
        }

    return pd.DataFrame({
        "term": [v for v in dunnett_model.group if v != "all factors"],
        "power": power_dict[power_type],
        "model": ["Dunnett"] * (len(dunnett_model.group) - 1),
        "n_rep": [dunnett_model.n_rep] * (len(dunnett_model.group) - 1)
    })