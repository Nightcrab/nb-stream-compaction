from numba import cuda, njit, prange
from numba.experimental import jitclass

from numba import int64, int32, int16, int8, boolean, deferred_type, optional, intp, void

from numba.typed import List

import time

import ctypes

import numba
import numpy as np


@njit([
    int8[:](int8[:]),
    int8[:](int16[:]),
    int8[:](int32[:]),
    int8[:](int8[:,:]),
    int8[:](int16[:,:]),
    int8[:](int32[:,:]),
    int8[:](int8[:,:,:]),
    int8[:](int16[:,:,:]),
    int8[:](int32[:,:,:]),
])
def any_mask(array):
    
    n = array.shape[0]

    mask = np.zeros(n, dtype="int8")

    for i in prange(n):
        mask[i] = np.any(array[i]) 

    return mask


@njit([
    int64[:](int8[:]),
    int64[:](int32[:]),
    int64[:](int64[:]),
])
def prefixsum(mask:int8[:]) -> int64[:]:

    out = np.cumsum(mask)
    out -= mask

    return out
   

@njit([
    void(int8[:], int8[:], int8[:], int64[:]),
    void(int8[:,:], int8[:,:], int8[:], int64[:]),
    void(int8[:,:,:], int8[:,:,:], int8[:], int64[:]),
    void(int16[:,:], int16[:,:], int8[:], int64[:]),
    void(int16[:,:,:], int16[:,:,:], int8[:], int64[:]),
    void(int32[:,:], int32[:,:], int8[:], int64[:]),
    void(int32[:,:,:], int32[:,:,:], int8[:], int64[:]),
])
def maskedfill(out, array, mask, idxs):
    for i in prange(array.shape[0]):
        idx = idxs[i]
        if array.ndim == 1:
            vec = array[i] if mask[i] else 0
        else:
            vec = array[i] if mask[i] else np.zeros(array.shape[1:], dtype=array.dtype)
        out[idx] = vec


@njit([
    int8[:](int8[:], int8[:]), 
    int8[:,:](int8[:,:], int8[:]), 
    int16[:,:](int16[:,:], int8[:]), 
    int32[:,:](int32[:,:], int8[:]), 
    int8[:,:,:](int8[:,:,:], int8[:]), 
    int16[:,:,:](int16[:,:,:], int8[:]), 
])
def compact(
    array, 
    mask
):

    length = array.shape[0]

    # prefix sum
    sums = prefixsum(mask)

    new_length = np.sum(mask)

    if array.ndim == 1:
        out = np.zeros((new_length,), dtype=array.dtype)
    else:
        out = np.zeros((new_length, *array.shape[1:]), dtype=array.dtype)

    maskedfill(out, array, mask, sums)

    return out 

