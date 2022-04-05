import numpy as np
import os
from itertools import chain


def _cupy_get_dev_attrs(dev):
    """
    Get select CUDA device attributes.
    Retrieve select attributes of the specified CUDA device that
    relate to maximum thread block and grid sizes.
    Parameters
    ----------
    d : cuda Device
        Device object to examine.
    Returns
    -------
    attrs : list
        List containing [MAX_THREADS_PER_BLOCK,
        (MAX_BLOCK_DIM_X, MAX_BLOCK_DIM_Y, MAX_BLOCK_DIM_Z),
        (MAX_GRID_DIM_X, MAX_GRID_DIM_Y, MAX_GRID_DIM_Z)]
    """
    attrs = dev.attributes
    return [attrs['MaxThreadsPerBlock'],
            (attrs['MaxBlockDimX'], attrs['MaxBlockDimY'], attrs['MaxBlockDimZ']),
            (attrs['MaxGridDimX'], attrs['MaxGridDimX'], attrs['MaxGridDimZ'])]


iceil = lambda n: int(np.ceil(n))


def select_block_grid_sizes(dev, data_shape, threads_per_block=None):
    """
    Determine CUDA block and grid dimensions given device constraints.
    Determine the CUDA block and grid dimensions allowed by a GPU
    device that are sufficient for processing every element of an
    array in a separate thread.
    Parameters
    ----------
    d : cuda Device
        Device object to be used.
    data_shape : tuple
        Shape of input data array. Must be of length 2.
    threads_per_block : int, optional
        Number of threads to execute in each block. If this is None,
        the maximum number of threads per block allowed by device `d`
        is used.
    Returns
    -------
    block_dim : tuple
        X, Y, and Z dimensions of minimal required thread block.
    grid_dim : tuple
        X and Y dimensions of minimal required block grid.
    Notes
    -----
    Using the scheme in this function, all of the threads in the grid can be enumerated
    as `i = blockIdx.y*max_threads_per_block*max_blocks_per_grid+
    blockIdx.x*max_threads_per_block+threadIdx.x`.
    For 2D shapes, the subscripts of the element `data[a, b]` where `data.shape == (A, B)`
    can be computed as
    `a = i/B`
    `b = mod(i,B)`.
    For 3D shapes, the subscripts of the element `data[a, b, c]` where
    `data.shape == (A, B, C)` can be computed as
    `a = i/(B*C)`
    `b = mod(i, B*C)/C`
    `c = mod(mod(i, B*C), C)`.
    For 4D shapes, the subscripts of the element `data[a, b, c, d]`
    where `data.shape == (A, B, C, D)` can be computed as
    `a = i/(B*C*D)`
    `b = mod(i, B*C*D)/(C*D)`
    `c = mod(mod(i, B*C*D)%(C*D))/D`
    `d = mod(mod(mod(i, B*C*D)%(C*D)), D)`
    It is advisable that the number of threads per block be a multiple
    of the warp size to fully utilize a device's computing resources.
    """

    import cupy as cp

    # Sanity checks:
    if np.isscalar(data_shape):
        data_shape = (data_shape,)

    # Number of elements to process; we need to cast the result of
    # np.prod to a Python int to prevent PyCUDA's kernel execution
    # framework from getting confused when
    N = int(np.prod(data_shape))

    # Get device constraints:
    max_threads_per_block, max_block_dim, max_grid_dim = _cupy_get_dev_attrs(dev)

    if threads_per_block is not None:
        if threads_per_block > max_threads_per_block:
            raise ValueError('threads per block exceeds device maximum')
        else:
            max_threads_per_block = threads_per_block

    # Actual number of thread blocks needed:
    blocks_needed = iceil(N/float(max_threads_per_block))
    if blocks_needed <= max_grid_dim[0]:
        return (max_threads_per_block, 1, 1), (blocks_needed, 1, 1)
    elif max_grid_dim[0] < blocks_needed <= max_grid_dim[0]*max_grid_dim[1]:
        return (max_threads_per_block, 1, 1), \
               (max_grid_dim[0], iceil(blocks_needed/float(max_grid_dim[0])), 1)
    elif max_grid_dim[0]*max_grid_dim[1] < blocks_needed <= max_grid_dim[0]*max_grid_dim[1]*max_grid_dim[2]:
        return (max_threads_per_block, 1, 1), \
               (max_grid_dim[0], max_grid_dim[1],
                iceil(blocks_needed/float(max_grid_dim[0]*max_grid_dim[1])))
    else:
        raise ValueError('array size too large')


_cuda_path_cache = 'NOT_INITIALIZED'
_optix_path_cache = 'NOT_INITIALIZED'


def get_path(key):
    env = os.environ.get(key, '')
    if env:
        return env.split(os.pathsep)
    else:
        return tuple()


def search_on_path(filenames, keys=('PATH',)):
    for p in chain(*[get_path(key) for key in keys]):
        for filename in filenames:
            full = os.path.abspath(os.path.join(p, filename))
            if os.path.exists(full):
                return os.path.abspath(full)
    return None


def get_cuda_path(environment_variable='CUDA_ROOT'):
    global _cuda_path_cache

    # Use a magic word to represent the cache not filled because None is a
    # valid return value.
    if _cuda_path_cache != 'NOT_INITIALIZED':
        return _cuda_path_cache

    nvcc_path = search_on_path(('nvcc', 'nvcc.exe'), keys=(environment_variable, 'PATH'))
    cuda_path_default = None
    if nvcc_path is not None:
        cuda_path_default = os.path.normpath(
            os.path.join(os.path.dirname(nvcc_path), '..'))

    if cuda_path_default is not None:
        _cuda_path_cache = cuda_path_default
    elif os.path.exists('/usr/local/cuda'):
        _cuda_path_cache = '/usr/local/cuda'
    else:
        _cuda_path_cache = None
    return _cuda_path_cache


def get_cuda_include_path(environment_variable='CUDA_ROOT'):
    cuda_path = get_cuda_path(environment_variable=environment_variable)
    if cuda_path is None:
        return None
    cuda_include_path = os.path.join(cuda_path, "include")
    if os.path.exists(cuda_include_path):
        return cuda_include_path
    else:
        return None