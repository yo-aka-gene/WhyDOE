import os
import subprocess
from tempfile import TemporaryDirectory
import yaml

import numpy as np
import pandas as pd
import polars as pl

from ._abstract import AbstractSimulator
from ._dunnett import Dunnett
from ._anova_power import sigma2


def mvtnorm_wrapper(
    arr: np.ndarray, 
    baseline: float, 
    n_rep: int, 
    sigma: float, 
    alpha: float
):
    tempdir = TemporaryDirectory()
    pl.from_numpy(arr).write_ipc(f"{tempdir.name}/arr.feather")
    cmd = f"Rscript {os.path.dirname(__file__)}/dunnett_power.R -t {tempdir.name} -m {baseline} -n {n_rep} -s {sigma} -a {alpha}"
    subprocess.call(cmd.split())
    # ret = pl.read_ipc(f"{tempdir.name}/power.feather", memory_map=False).to_numpy()
    with open(f"{tempdir.name}/power.yaml") as f:
        ret = yaml.safe_load(f)
    tempdir.cleanup()
    return ret


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
        power_dict = mvtnorm_wrapper(
            arr=(dunnett_model.coef - dunnett_model.baseline).y.values,
            baseline=dunnett_model.baseline.y,
            n_rep=dunnett_model.n_rep,
            sigma=np.sqrt(sigma2(dunnett_model.simulation).item()),
            alpha=alpha
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