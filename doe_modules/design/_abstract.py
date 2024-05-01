from itertools import combinations

import numpy as np
import pandas as pd


class DesignMatrix:
    def __init__(self, matrix: np.ndarray):
        self.matrix = pd.DataFrame(
            matrix,
            index = [f"#{i + 1}" for i in range(matrix.shape[0])],
            columns = [f"X{i + 1}" for i in range(matrix.shape[1])]
        )


    def __call__(self, binarize: bool = False):
        return ((self.matrix + 1) / 2).astype(bool) if binarize else self.matrix


    def interactions(self):
        return pd.concat(
            [
                self.matrix,
                pd.DataFrame({
                    f"{v[0]}{v[1]}": self.matrix.loc[
                        :, v
                    ].prod(axis=1) for v in combinations(
                        self.matrix.columns, 2
                    )
                })
            ],
            axis=1
        )


class DOE:
    def __init__(self):
        self.is_initialized = True
