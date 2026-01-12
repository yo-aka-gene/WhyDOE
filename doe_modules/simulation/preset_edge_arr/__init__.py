import numpy as np


## the most PB-suited
model_pi_arr = np.array([
    1, 1, 1, 1, 1, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0,
    0, 0,
    0
])


## the most C+LOO-suited
model_delta_arr = np.array([
    0, 0, 0, 0, 0, 0, 0, 0, 1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1,
    1, 1, 1, 1,
    1, 1, 1,
    1, 1,
    1
])


## the PB-suited ver 2
model_sigma_arr = np.array([
    1, 1, 1, 1, 0, 0, 0, 0, -1,
    1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1,
    1, 1, 1, 1,
    1, 1, 1,
    1, 1,
    1
])


__all__ = [
    "model_delta_arr",
    "model_pi_arr",
    "model_sigma_arr"
]
