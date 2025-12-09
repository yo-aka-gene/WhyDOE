import os

import numpy as np
import pandas as pd
import rpy2.robjects as ro
import rpy2.robjects.vectors as ro_vectors
from rpy2.robjects.packages import importr, isinstalled
from rpy2.robjects import numpy2ri

from ._abstract import DesignMatrix, DOE
from ._fullfact import FullFactorial


numpy2ri.activate()

utils = importr('utils')
base = importr('base')
packages = ["AlgDesign"]
repos = "https://cloud.r-project.org/"
rscript = f"{os.path.dirname(__file__)}/d_optimization.R"
func_name = "d_optimize_core"
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


def d_optimize(
    dsmatrix: DesignMatrix,
    n_add: int = 0,
    n_total: int = None,
    random_state: int = 0
) -> np.ndarray:
    if n_add == 0 and n_total is not None and isinstance(n_total, int):
        assert n_total >= dsmatrix.shape[0], \
            f"Invalid n_total value: n_total={n_total} should be an integer larger than the original number of trials (>={dsmatrix.shape[0]})"
        n_add = n_total - dsmatrix.shape[0]
    assert isinstance(n_add, (int, np.int64)) and n_add >= 0, \
        f"Invalid n_add value: n_add={n_add} should be an integer >=0"

    n_factor = dsmatrix.shape[1]

    _initialize_r_func()

    return np.asarray(
        R_FUNC(
            dsmatrix=dsmatrix.values,
            candidate=FullFactorial().get_exmatrix(n_factor).values,
            n_add=n_add,
            random_state=random_state
        )
        # NOTE: The following R call involves cross-language matrix conversion (Python/NumPy C-Major -> R Fortran-Major).
        # R (AlgDesign) expects the standard (Trials x Factors) structure.
        # However, due to the difference in internal memory ordering (Column-Major vs. Row-Major) during rpy2's conversion process,
        # the matrix returned to Python is interpreted as TRANSPOSED (Factors x Trials).
    ).T # FINAL CORRECTION: .T reverses the unintended transposition caused by the R-Python memory layout mismatch.

#T he final .T operation reverses the unintended transposition that occurs during the R-to-Python conversion. 
# The underlying cause is the conflict between R's internal # Column-Major memory order (Fortran standard) and NumPy's C-Major (Row-Major) standard, 
# which leads to the output being interpreted as transposed ($P \times N$) upon # reading back into Python. 
# This is necessary because, while R correctly performs the optimization on a Trials $\times$ Factors structure, the data must be physically 
# manipulated to satisfy both memory models.


class DOptimization(DOE):
    def __init__(self, base: DOE, name: str = None):
        super().__init__(name="D-optimized " + base().name if name is None else name)
        self.base = base


    def get_exmatrix(
        self, 
        n_factor: int,
        n_add: int = 0,
        n_total: int = None,
        random_state: int = 0,
        **kwargs
    ) -> DesignMatrix:
        optimized = d_optimize(
            self.base().get_exmatrix(n_factor), 
            n_add=n_add, n_total=n_total,
            random_state=random_state
        )
        self.title = f"{self.name} design with {len(optimized)} trials (n={n_factor})"
        return DesignMatrix(optimized)


    def __call__(self):
        return super().__call__()


def d_criterion(X: np.ndarray) -> float:
    return np.linalg.det(np.linalg.inv(X.T @ X))