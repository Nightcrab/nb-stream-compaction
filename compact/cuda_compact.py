from numba import cuda, njit, prange
from numba.experimental import jitclass

from numba import int64, int32, int16, int8, boolean, deferred_type, optional, intp, void

from numba.typed import List

import time

import ctypes

import numba
import numpy as np

import cupy

import warmup

from cpu_compact import prefixsum, cpu_compact, any_mask

import cuda_utils


WARP_SIZE = 32


def _cupy_prefixsum(mask: cupy.ndarray):
    out = cupy.cumsum(mask)
    out -= mask
    return out

@cuda.jit([
    int8(int8[:]),
    int8(int16[:]),
    int8(int32[:]),
    int8(int8[:,:]),
    int8(int16[:,:]),
    int8(int32[:,:]),
], cache=True, device=True)
def any(d_array):
    ret = False
    for i in range(d_array.shape[0]):
        if d_array.ndim == 2:
            for j in range(d_array.shape[1]):
                if d_array[i][j] != 0:
                    ret = True
        elif d_array[i] != 0:
            ret = True

    return ret


@cuda.jit([
    void(int8[:, :], int8[:, :]),
    void(int16[:, :], int8[:, :]),
    void(int32[:, :], int8[:, :]),
], cache=True)
def _any_mask(d_array, d_mask):

    idx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x

    if idx < d_array.shape[0]:
        d_mask[idx] = any(d_array[idx])


@cuda.jit([
    void(int8[:], int64[:], int8[:, :]),
    void(int8[:, :], int64[:], int8[:, :]),
    void(int8[:, :, :], int64[:], int8[:, :]),
    void(int16[:, :], int64[:], int8[:, :]),
    void(int16[:, :, :], int64[:], int8[:, :]),
    void(int32[:, :], int64[:], int8[:, :]),
    void(int32[:, :, :], int64[:], int8[:, :])
])
def block_offsets(d_array, b_offsets, d_predicate):

    idx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x

    t_id = cuda.threadIdx.x
    b_id = cuda.blockIdx.x

    if (idx < d_array.shape[0]):
        predicate = d_predicate[idx][0]
        # population of block
        count = cuda.syncthreads_count(predicate)
        if t_id == 0:
            b_offsets[b_id] = count


@cuda.jit([
    void(int8[:], int8[:], int64[:], int8[:, :]),
    void(int8[:, :], int8[:, :], int64[:], int8[:, :]),
    void(int16[:, :], int16[:, :], int64[:], int8[:, :]),
    void(int32[:, :], int32[:, :], int64[:], int8[:, :]),
    void(int8[:, :, :], int8[:, :, :], int64[:], int8[:, :]),
], cache=True)
def _compact(d_array, d_out, b_offsets, d_predicate):

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

        predicate = d_predicate[idx][0]

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

            cuda_utils.setitem(d_out, target_idx, d_array[idx])


def _np_compact(
    array, 
    mask: optional(int8[:, :]) = None,
):
    array_length = array.shape[0]

    block_size = 64
    blocks = -(array_length // -block_size)

    offsets = np.zeros((blocks,), dtype="int64")

    d_array = cuda.to_device(array)

    b_offsets = cuda.device_array((blocks,), dtype="int64")

    if mask is None:
        d_mask = cuda.device_array((array_length, 1), dtype="int8")
        _any_mask[blocks, block_size](d_array, d_mask)
    else:
        mask = np.reshape(mask, (mask.shape[0], 1))
        d_mask = cuda.to_device(mask)

    block_offsets[blocks, block_size](d_array, b_offsets, d_mask)

    b_offsets.copy_to_host(offsets)
    
    length = np.sum(offsets)

    d_out = cuda.device_array((length, *array.shape[1:]), dtype=array.dtype)

    offsets = prefixsum(offsets)

    b_offsets = cuda.to_device(offsets)

    _compact[blocks, block_size](d_array, d_out, b_offsets, d_mask)

    out = np.empty_like(d_out, dtype=array.dtype)

    # move data to host
    d_out.copy_to_host(out)
    b_offsets.copy_to_host(offsets)

    return out

@profile
def _device_compact(
    d_array, 
    d_mask,
):
    array_length = d_array.shape[0]

    block_size = 512
    blocks = -(array_length // -block_size)

    b_offsets = cuda.device_array((blocks,), dtype="int64")

    if d_mask is None:
        d_mask = cuda.device_array((array_length, 1), dtype="int8")
        _any_mask[blocks, block_size](d_array, d_mask)
    else:
        d_mask = d_mask.reshape((d_mask.shape[0], 1))

    block_offsets[blocks, block_size](d_array, b_offsets, d_mask)

    if d_array.size < 1000000:
        # numpy scan
        offsets = np.empty((blocks,), dtype="int64")
        b_offsets.copy_to_host(offsets)
        length = np.sum(offsets)
        offsets = prefixsum(offsets)
        b_offsets = cuda.to_device(offsets)
    else:
        # cupy scan
        cupy_b_offsets = cupy.asarray(b_offsets)
        b_offsets_psum = _cupy_prefixsum(cupy_b_offsets)
        length = int(b_offsets_psum[-1] + b_offsets[-1])
        b_offsets = b_offsets_psum

    d_out = cuda.device_array((length, *d_array.shape[1:]), dtype=d_array.dtype)

    _compact[blocks, block_size](d_array, d_out, b_offsets, d_mask)

    return d_out


def gpu_compact(
    array, 
    mask: optional(int8[:, :]) = None,
):
    if cuda.is_cuda_array(array):
        return _device_compact(array, mask)
    elif isinstance(array, np.ndarray):
        return _np_compact(array, mask)
    else:
        raise Exception("gpu_compact: input must be array or device array")


def test(n):
    x = np.array([[1],[0],[2],[0],[3],[4],[5],[0],[0],[6]], dtype="int8")

    x = np.repeat(x, n // 10, axis=0)

    d_x = cuda.to_device(x)

    y = gpu_compact(d_x)

    t = time.perf_counter()
    for i in range(10):
        y = gpu_compact(d_x)
    print("total execution time (gpu): " + str((time.perf_counter() - t) * 1000 / 10) + "ms")


    t = time.perf_counter()
    z = cpu_compact(x, mask=any_mask(x))
    print("total execution time (cpu): " + str((time.perf_counter() - t) * 1000) + "ms")

    assert np.allclose(np.array(y), z)
