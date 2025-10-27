import numpy as np
import pandas as pd
from scipy.stats import f, ncf
import statsmodels.api as sm

from ._abstract import AbstractSimulator


def X(simulator: AbstractSimulator) -> np.ndarray:
    return sm.add_constant(simulator.exmatrix.values)


def Y(simulator: AbstractSimulator) -> np.ndarray:
    return simulator.exresult.reshape(-1, 1)


def P(matrix: np.ndarray) -> np.ndarray:
    return matrix @ np.linalg.inv(matrix.T @ matrix) @ matrix.T


def Xi(simulator: AbstractSimulator, i: int) -> np.ndarray:
    return simulator.exmatrix.values[:, i].reshape(-1, 1)


def Xr(simulator: AbstractSimulator, i: int) -> np.ndarray:
    return sm.add_constant(
        simulator.exmatrix.values[
            :, 
            [idx for idx in range(simulator.n_factor) if idx != i]
        ]
    )


def Qr(simulator: AbstractSimulator, i: int) -> np.ndarray:
    n_max = len(X(simulator))
    return np.eye(n_max) - P(Xr(simulator, i))


def phi_i(simulator: AbstractSimulator, i: int) -> np.int:
    return np.linalg.matrix_rank(Qr(simulator, i) @ Xi(simulator, i))


def SSi(simulator: AbstractSimulator, i: int) -> np.ndarray:
    return Y(simulator).T @ P(Qr(simulator, i) @ Xi(simulator, i)) @ Y(simulator)


def phi_e(simulator: AbstractSimulator) -> np.int:
    n_max = len(X(simulator))
    return n_max - np.linalg.matrix_rank(X(simulator))


def SSe(simulator: AbstractSimulator) -> np.ndarray:
    n_max = len(X(simulator))
    return Y(simulator).T @ (np.eye(n_max) - P(X(simulator))) @ Y(simulator)


def sigma2(simulator: AbstractSimulator) -> np.ndarray:
    """
    unbiased estimator of $\sigma^2$ is $\frac{SS_e}{\phi_e}$
    """
    return SSe(simulator) / phi_e(simulator)


def norm2(vector: np.ndarray) -> np.float:
    """
    function to return a norm squared
    """
    return vector.T @ vector


def beta(X_des: np.ndarray, y_obs: np.ndarray) -> np.ndarray:
    """
    estimates a coefficient vector from given design matrix X_des and observation vector y_obs
    """
    return np.linalg.inv(X_des.T @ X_des) @ X_des.T @ y_obs


def lambda_i(simulator: AbstractSimulator, i: int) -> np.float:
    """
    estimates noncentrality parameter $\lambda_i$
    """
    return norm2(
        Qr(simulator, i) @ Xi(simulator, i) @ beta(Xi(simulator, i), Y(simulator))
    ) / sigma2(simulator)


def Fi(simulator: AbstractSimulator, i: int) -> np.float:
    return (
        SSi(simulator, i) / phi_i(simulator, i)
    ) / sigma2(simulator)


def power(
    simulator: AbstractSimulator, 
    i: int, 
    alpha: float = 0.05
) -> np.float:
    """
    calculate $1-\beta$ for the given factor
    """
    return ncf.sf(
        x=f.isf(
            q=alpha, dfn=phi_i(simulator, i), dfd=phi_e(simulator)
        ),
        dfn=phi_i(simulator, i), dfd=phi_e(simulator),
        nc=lambda_i(simulator, i)
    )


def anova_power(simulator: AbstractSimulator) -> pd.DataFrame:
    """
    returns summary DataFrame
    """
    term = simulator.metadata["factor_list"]
    n = simulator.n_factor
    return pd.DataFrame({
        "term": term,
        "power": np.vectorize(lambda i: power(simulator, i))(np.arange(n)),
        "model": [simulator.metadata["design"] for _ in term],
        "n_rep": [simulator.metadata["n_rep"] for _ in term],
    })
