from numba import cuda, njit, prange
from numba.experimental import jitclass

from numba import int64, int32, int16, int8, boolean, deferred_type, optional, intp, void

from numba.typed import List

import time

import ctypes

import numba
import numpy as np

import warmup

from cpu_compact import prefixsum, compact, any_mask


WARP_SIZE = 32


@cuda.jit([
    int8(int8[:]),
    int8(int16[:]),
    int8(int32[:]),
    int8(int8[:,:]),
    int8(int16[:,:]),
    int8(int32[:,:]),
], device=True)
def any(array):
    ret = False
    for i in range(array.shape[0]):
        if array.ndim == 2:
            for j in range(array.shape[1]):
                if array[i][j] != 0:
                    ret = True
        elif array[i] != 0:
            ret = True
    return ret


@cuda.jit([
    void(int8[:, :], int64[:])
])
def block_offsets(d_array, b_offsets):

    idx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x

    t_id = cuda.threadIdx.x
    b_id = cuda.blockIdx.x

    if (idx < d_array.shape[0]):
        predicate = any(d_array[idx])
        # population of block
        count = cuda.syncthreads_count(predicate)
        if t_id == 0:
            b_offsets[b_id] = count


@cuda.jit([
    void(int8[:, :], int8[:, :], int64[:]),
    void(int16[:, :], int16[:, :], int64[:]),
    void(int32[:, :], int32[:, :], int64[:]),
    void(int8[:, :, :], int8[:, :, :], int64[:]),
])
def _compact(d_array, d_out, b_offsets):

    idx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x

    t_id = cuda.threadIdx.x
    b_id = cuda.blockIdx.x

    bsize = cuda.blockDim.x
    
    warps = bsize // WARP_SIZE

    # number of true threads preceding each warp
    warp_counts = cuda.shared.array(32, dtype=int32)

    if (idx < d_array.shape[0]):

        w_id = t_id // WARP_SIZE

        lane = t_id % WARP_SIZE

        predicate = any(d_array[idx])

        prefix_mask = ~(~0 << lane)

        # predicate of prefix (in warp)
        prefix_ballot = cuda.ballot_sync(prefix_mask, predicate)

        # predicate of warp
        thread_ballot = cuda.ballot_sync(~0, predicate)

        # prefix population
        t_pop = cuda.popc(prefix_ballot)

        # warp population
        w_pop = cuda.popc(thread_ballot)

        # store warp pop in shared array for summing
        if lane == WARP_SIZE - 1:
            warp_counts[w_id] = w_pop

        cuda.syncthreads()

        warps_mask = ~(~0 << warps)

        # binary scan over all warps
        if (w_id == 0 and lane < warps):

            w_sum = 0

            for i in range(6):
                b_j = cuda.ballot_sync(warps_mask, warp_counts[lane] & (1 << i))
                w_sum += cuda.popc(b_j & prefix_mask) << i
            
            warp_counts[lane] = w_sum

        cuda.syncthreads()
        
        if predicate:
            target_idx = t_pop + warp_counts[w_id] + b_offsets[b_id]

            if d_array.ndim == 1:
                d_out[target_idx] = d_array[idx]
            else:
                for i in range(d_array.shape[1]):
                    if d_array.ndim == 3:
                        for j in range(d_array.shape[2]):
                            d_out[target_idx][i][j] = d_array[idx][i][j]
                    else:
                        d_out[target_idx][i] = d_array[idx][i]



def gpu_compact(array):

    # host spends 6 ms on an array of size 10,000,000
    # 2 ms to launch kernel

    block_size = 64
    blocks = -(array.shape[0] // -block_size)

    out = np.empty_like(array, dtype=array.dtype)

    offsets = np.zeros((blocks,), dtype="int64")

    # allocate device arrays
    d_array = cuda.to_device(array)
    d_out = cuda.device_array_like(array)
    b_offsets = cuda.device_array((blocks,), dtype="int64")

    t = time.time()

    block_offsets[blocks, block_size](d_array, b_offsets)

    print("kernel 1 execution time: " + str((time.time() - t) * 1000) + "ms")

    b_offsets.copy_to_host(offsets)
    
    length = np.sum(offsets)

    offsets = prefixsum(offsets)

    b_offsets = cuda.to_device(offsets)

    t = time.time()

    _compact[blocks, block_size](d_array, d_out, b_offsets)

    print("kernel 2 execution time: " + str((time.time() - t) * 1000) + "ms")

    # move data to host
    d_out.copy_to_host(out)
    b_offsets.copy_to_host(offsets)

    # deallocate device array
    d_out = None

    print("length", length)

    shape = (length, *out.shape[1:])

    # resize to ignore trailing zeros
    out.resize(shape)

    return out

def test():
    x = np.array([[1],[0],[2],[0],[3],[4],[5],[0],], dtype="int8")

    x = np.repeat(x, 2000000000, axis=0)

    t = time.time()
    y = gpu_compact(x)
    print("total execution time (gpu): " + str((time.time() - t) * 1000) + "ms")

    t = time.time()
    z = compact(x, mask=any_mask(x))
    print("total execution time (cpu): " + str((time.time() - t) * 1000) + "ms")


    assert np.allclose(y, z)