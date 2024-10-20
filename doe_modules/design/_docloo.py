from ._abstract import DesignMatrix
from ._cloo import CLOO
from ._d_optimization import DOptimization, d_criterion

class DOCLOO(DOptimization):
    def __init__(self):
        super().__init__(base=CLOO, name="DO-C+LOO")

    def get_exmatrix(
        self, 
        n_factor: int,
        n_add: int = 0,
        n_total: int = None,
        # sort: bool = False,
        **kwargs
    ) -> DesignMatrix:
        return super().get_exmatrix(
            n_factor=n_factor, n_add=n_add, n_total=n_total,
            # sort=sort, 
            **kwargs
        )