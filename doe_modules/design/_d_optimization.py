import os
import subprocess
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import polars as pl

from ._abstract import DesignMatrix, DOE
from ._fullfact import FullFactorial


def algdesign_wrapper(arr: np.ndarray, candidate: np.ndarray, n_add: int) -> np.ndarray:
    tempdir = TemporaryDirectory()
    pl.from_numpy(arr).write_ipc(f"{tempdir.name}/arr.feather")
    pl.from_numpy(candidate).write_ipc(f"{tempdir.name}/candidate.feather")
    cmd = f"Rscript {os.path.dirname(__file__)}/d_optimization.R -t {tempdir.name} -a {n_add}"
    subprocess.call(cmd.split())
    ret = pl.read_ipc(f"{tempdir.name}/opt.feather", memory_map=False).to_numpy()
    tempdir.cleanup()
    return ret


def d_optimize(
    dsmatrix: DesignMatrix,
    n_add: int = 0,
    n_total: int = None
) -> np.ndarray:
    if n_add == 0 and n_total is not None and isinstance(n_total, int):
        assert n_total >= dsmatrix.shape[0], \
            f"Invalid n_total value: n_total={n_total} should be an integer larger than the original number of trials (>={dsmatrix.shape[0]})"
        n_add = n_total - dsmatrix.shape[0]
    assert isinstance(n_add, int) and n_add >= 0, \
        f"Invalid n_add value: n_add={n_add} should be an integer >=0"

    n_factor = dsmatrix.shape[1]

    return algdesign_wrapper(
        arr=dsmatrix.values,
        candidate=FullFactorial().get_exmatrix(n_factor).values,
        n_add=n_add
    )


# def matsort(arr: np.ndarray, axis: int) -> np.ndarray:
#     return arr[np.argsort(arr.sum(axis=1)),:] if axis == 0 else arr[:,np.argsort(arr.sum(axis=0))]


# def format_optarr(opt: np.ndarray, org: np.ndarray) -> np.ndarray:
#     added = opt[len(org):]
#     return np.vstack([org, matsort(matsort(added, axis=1), axis=0)])


class DOptimization(DOE):
    def __init__(self, base: DOE, name: str = None):
        super().__init__(name="D-optimized " + base().name if name is None else name)
        self.base = base


    def get_exmatrix(
        self, 
        n_factor: int,
        n_add: int = 0,
        n_total: int = None,
        # sort: bool = True,
        **kwargs
    ) -> DesignMatrix:
        opt = d_optimize(
            self.base().get_exmatrix(n_factor), 
            n_add=n_add, n_total=n_total
        )
        self.title = f"{self.name} design with {len(opt)} trials (n={n_factor})"
        return DesignMatrix(
            opt
            # format_optarr(opt, self.base().get_exmatrix(n_factor).values) if sort else opt
        )


    def __call__(self):
        return super().__call__()


def d_criterion(X: np.ndarray) -> float:
    return np.linalg.det(np.linalg.inv(X.T @ X))