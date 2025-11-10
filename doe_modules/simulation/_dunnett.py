import numpy as np
import pandas as pd
from scipy.stats import dunnett

from doe_modules.simulation import AbstractSimulator
from doe_modules.design import DesignMatrix
from doe_modules.preferences.pvalues import sign, asterisk, p_format


class Dunnett:
    def __init__(
        self, 
        simulation: AbstractSimulator,
    ):
        assert simulation.metadata["design"] == "C+LOO", \
            f"Experimental design should be C+LOO, got {simulation.metadata['design']}"
        self.simulation = simulation
        self.n_rep = simulation.metadata["n_rep"]
        self.group = ["all factors"] + simulation.metadata["factor_list"]
        self.result = pd.DataFrame({
            "group": self.group * self.n_rep,
            "y": simulation.exresult
        })
        
    def summary(
        self,
        dtype: type = str,
        alpha: float = .05,
        test_kwargs: dict = {"alternative": "two-sided", "random_state": 0}
    ):
        if self.n_rep > 1:
            self.coef = self.result.groupby("group").mean().loc[
                [v for v in self.group if v != "all factors"]
            ]
            self.baseline = self.result.groupby("group").mean().loc["all factors"]                       
            self.pvals = dunnett(
                *[self.result[self.result.group == g].y for g in [v for v in self.group if v != "all factors"]],
                control=self.result[self.result.group == "all factors"].y,
                **test_kwargs
            ).pvalue
            interpretation = (
                (
                    2 * (self.baseline > self.coef).astype(int) - 1
                ) * (self.pvals < alpha).reshape(-1, 1).astype(int)
            )
            interpretation = interpretation.applymap(
                lambda i: {0: "N.S", 1: "Up.", -1: "Down."}[i]
            ) if dtype == str else interpretation
            
            ret = interpretation.y
            ret.name = None
            ret.index.name = None
            
            return ret
        
        else:
            return pd.Series(
                [np.nan] * (len(self.group) - 1),
                index=[v for v in self.group if v != "all factors"]
            )