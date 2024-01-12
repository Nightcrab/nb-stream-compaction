from numba import cuda
import numpy as np

# warm-up cuda
cuda.to_device(np.empty(0))