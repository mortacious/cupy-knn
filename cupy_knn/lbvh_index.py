import cupy as cp
import numpy as np
import pathlib
from typing import Optional
from .cuda_util import select_block_grid_sizes, get_cuda_include_path

_cuda_include = get_cuda_include_path()

_file_path = pathlib.Path(__file__).parent

with open(_file_path / "cuda/lbvh_kernels.cu", 'r') as f:
    _lbvh_src = f.read()

_construct_tree_kernels = cp.RawModule(code=_lbvh_src,
                                       options=('--std=c++11', f' -I{pathlib.Path(__file__).parent / "cuda"}',
                                                f'-I{_cuda_include}', '--use_fast_math', '--extra-device-vectorization'),
                                       name_expressions=('compute_morton_kernel',
                                                         'compute_morton_points_kernel',
                                                         'initialize_tree_kernel',
                                                         'construct_tree_kernel',
                                                         'optimize_tree_kernel',
                                                         'compute_free_indices_kernel',
                                                         'compact_tree_kernel'))

with open(_file_path / "cuda/query_knn_kernels.cu", 'r') as f:
    _query_knn_src = f.read()


class LBVHIndex(object):
    """
    A fast left-balanced BVH-tree based nearest neighbor index for 3-D data.

    Parameters
    ----------
    leaf_size: int
        The maximum leaf size in the tree

    sort_threshold: int
        Threshold, above which the query points will be sorted in morton order prior
        to passing them to the knn query kernel to prevent warp divergence.

    compact: bool
        Compact the tree after optimization by removing the freed space in between. This is done in-place by default.
        If the leaf_size is 1, this parameter is ignored.

    shrink_to_fit: bool
        If the tree is compacted, shrink it's used buffer to fit the new size.
        This requires a full copy, which might not be possible on the gpu depending on the index size.
        If compact is False this parameter is ignored.
    """
    def __init__(self, leaf_size: int = 32, sort_threshold: int = 10_000, compact=True, shrink_to_fit=False):
        self.num_objects = -1
        self.num_nodes = -1
        self.leaf_size = leaf_size
        self._sort_threshold = sort_threshold
        self.compact = compact
        self.shrink_to_fit = shrink_to_fit

        self.points = None
        self.sorted_indices = None
        self.K = None
        self.radius = None
        self.extent = None
        self.nodes = None
        self.root_node = None

        self.tree_dtype = cp.dtype({"names": ("aabb", "parent", "child_left", "child_right", "atomic", "range_left", "range_right"),
                       "formats": ("6f4", "u4", "u4", "u4", "i4", "u4", "u4")}, align=True)
        self._query_knn_kernels = None
        self._query_knn_kernel = None

        self._compute_morton_kernel_float = _construct_tree_kernels.get_function('compute_morton_kernel')
        self._compute_morton_points_kernel_float = _construct_tree_kernels.get_function('compute_morton_points_kernel')
        self._construct_tree_kernel_float = _construct_tree_kernels.get_function('construct_tree_kernel')
        self._initialize_tree_kernel_float = _construct_tree_kernels.get_function('initialize_tree_kernel')
        self._optimize_tree_kernel_float = _construct_tree_kernels.get_function('optimize_tree_kernel')
        self._compute_free_indices_kernel = _construct_tree_kernels.get_function('compute_free_indices_kernel')
        self._compact_tree_kernel = _construct_tree_kernels.get_function('compact_tree_kernel')

    def prepare_knn(self, k: int, radius: Optional[float] = None):
        """
        Prepare the index for KNN search with the specified k and maximum radius.

        Parameters
        ----------
        k: int
            The number of nearest neighbors to search.
        radius: float
            The maximum radius to search the neighbors inside. A value of None will result in an infinite radius.
        """
        self.K = k
        if radius is None:
            radius = cp.finfo(cp.float32).max
        else:
            radius = radius**2

        self.radius = radius
        self._query_knn_kernels = cp.RawModule(code=_query_knn_src,
                                               options=('--std=c++11', f' -I{pathlib.Path(__file__).parent / "cuda"}',
                                                        f'-DK={self.K}', f'-I{_cuda_include}', '--use_fast_math',
                                                        '--extra-device-vectorization'),
                                               name_expressions=('query_knn_kernel',))

        self._query_knn_kernel = self._query_knn_kernels.get_function('query_knn_kernel')

    def build(self, points: cp.ndarray):
        """
        Build the search index using the given points. The point array is first sorted in morton order and afterwards
        the search tree will be built in a bottom up fashion. The resulting simple tree is then further optimized
        by accumulating leafs until the maximum leaf size is reached.

        Parameters
        ----------
        points: cp.ndarray of shape (n, 3)
            The 3-D points to build the index from.
        """
        self.points = cp.ascontiguousarray(cp.asarray(points, dtype=cp.float32)).reshape(-1, 3)

        assert len(self.points.shape) == 2 and self.points.shape[1] == 3, "Only 3-D points supported"

        self.num_objects = self.points.shape[0]
        self.num_nodes = self.num_objects*2 - 1

        # init aabbs
        aabbs = cp.empty((self.num_objects, 2, 3), dtype=points.dtype)
        aabbs[:, 0] = self.points  # initialize the bounding boxes of the leafs
        aabbs[:, 1] = self.points

        min = cp.min(self.points, axis=0)
        max = cp.max(self.points, axis=0)
        self.extent = cp.stack((min, max), axis=0)

        # compute the morton codes of the aabbs
        morton_codes = cp.empty(self.num_objects, dtype=cp.uint64)

        block_dim, grid_dim = select_block_grid_sizes(aabbs.device, aabbs.shape[0])

        self._compute_morton_kernel_float(grid_dim, block_dim, (aabbs, self.extent, morton_codes, aabbs.shape[0]))

        # sort everything by the morton codes
        self.sorted_indices = cp.argsort(morton_codes).astype(cp.uint32)

        morton_codes = morton_codes[self.sorted_indices]
        aabbs = aabbs[self.sorted_indices]

        # allocate space for the nodes as a raw cuda array
        self.nodes = cp.cuda.alloc(self.tree_dtype.itemsize * self.num_nodes)

        root_node = cp.full((1,), cp.iinfo(np.uint32).max, dtype=cp.uint32)

        self._initialize_tree_kernel_float(grid_dim, block_dim, (self.nodes,
                                                                 aabbs,
                                                                 self.num_objects))
        self._construct_tree_kernel_float(grid_dim, block_dim, (self.nodes,
                                                                root_node,
                                                                morton_codes,
                                                                self.num_objects))
        if self.leaf_size > 1:
            valid = cp.full(self.num_nodes, 1, dtype=cp.uint32)
            self._optimize_tree_kernel_float(grid_dim, block_dim, (self.nodes,
                                                                   root_node,
                                                                   valid,
                                                                   cp.uint32(self.leaf_size),
                                                                   self.num_objects))
            # compact the tree to increase bandwidth
            if self.compact:

                # compute the prefix sum of the valid array to determine the indices of the free space
                valid_sums = cp.zeros(self.num_nodes+1, dtype=cp.uint32)
                # allocate the one element larger than the number of nodes
                # cumsum must start with 0 here so leave out the first element
                cp.cumsum(valid, out=valid_sums[1:])

                # get the number of actually used nodes after optimization
                new_node_count = valid_sums[self.num_nodes].get()
                # leave out the last element again to align with the valid array

                valid_sums_aligned = valid_sums[:-1]

                # compute the isum parameter to get the indices of the free elements
                isum = (cp.arange(self.num_nodes, dtype=cp.uint32)-valid_sums_aligned)
                free_indices_size = int(isum[new_node_count].get())  # number of free elements in the optimized tree array

                free = valid[:free_indices_size]  # reuse the valid space as it is not needed any more

                # compute the free indices
                block_dim, grid_dim = select_block_grid_sizes(isum.device, new_node_count)
                self._compute_free_indices_kernel(grid_dim, block_dim, (valid_sums, isum, free, new_node_count))

                # get the sum of the first object that has to be moved
                first_moved = valid_sums[new_node_count].get()

                block_dim, grid_dim = select_block_grid_sizes(isum.device, self.num_nodes)

                self._compact_tree_kernel(grid_dim, block_dim, (self.nodes, root_node, valid_sums_aligned, free, first_moved, new_node_count, self.num_nodes))

                if self.shrink_to_fit:
                    nodes_old = self.nodes
                    self.nodes = cp.cuda.alloc(self.tree_dtype.itemsize * new_node_count)
                    self.nodes.copy_from_device(nodes_old, self.tree_dtype.itemsize * new_node_count)

                self.num_nodes = new_node_count

        # fetch to root node's location in the tree
        self.root_node = int(root_node.get()[0])

    def query_knn(self, queries: cp.ndarray):
        """
        Query the search tree for the nearest neighbors of the query points. 'prepare_knn' must have been
        called prior to colling this function to prepare the cuda kernels for the specified number of neighbors.

        Parameters
        ----------
        queries: cp.ndarray of shape (n, 3)
            The query points to search the nearest neighbors.

        Returns
        -------
        indices: cp.ndarray of shape (n, K)
            The indices of the nearest neighbors
        distances: cp.ndarray of shape (n, K)
            The squared distances of the nearest neighbors towards their query point
        n_neighbors: cp.ndarray of shape (n,)
            The number of nearest neighbors found

        """
        if self.num_nodes < 0:
            raise ValueError("Index has not been built yet. Call 'build' first.")
        if self.K is None:
            raise ValueError("Index has not been prepared for knn query. Use 'prepare_knn' first.")

        queries = cp.ascontiguousarray(cp.asarray(queries, dtype=cp.float32)).reshape(-1, 3)

        stream = cp.cuda.get_current_stream()

        # only for large queries: sort them in morton order to prevent too much warp divergence on tree traversal
        if queries.shape[0] > self._sort_threshold:
            morton_codes = cp.empty(queries.shape[0], dtype=cp.uint64)
            block_dim, grid_dim = select_block_grid_sizes(queries.device, queries.shape[0])
            self._compute_morton_points_kernel_float(grid_dim, block_dim, (queries, self.extent, morton_codes, queries.shape[0]))
            sorted_indices = cp.argsort(morton_codes)
            queries = queries[sorted_indices]
            stream.synchronize()

        # use the maximum allowed threads per block from the kernel (depends on the number of registers)
        max_threads_per_block = self._query_knn_kernel.attributes['max_threads_per_block']
        block_dim, grid_dim = select_block_grid_sizes(queries.device, queries.shape[0], threads_per_block=max_threads_per_block)
        indices_out = cp.full((queries.shape[0], self.K), cp.iinfo(cp.uint32).max, dtype=cp.uint32)
        distances_out = cp.full((queries.shape[0], self.K), cp.finfo(cp.float32).max, dtype=cp.float32)
        n_neighbors_out = cp.zeros((queries.shape[0]), dtype=cp.uint32)

        self._query_knn_kernel(grid_dim, block_dim, (self.nodes,
                                                     self.points,
                                                     self.sorted_indices,
                                                     cp.uint32(self.root_node),
                                                     cp.float32(self.radius),
                                                     queries,
                                                     indices_out,
                                                     distances_out,
                                                     n_neighbors_out,
                                                     queries.shape[0]))
        stream.synchronize()

        if queries.shape[0] > self._sort_threshold:
            indices_tmp = cp.empty_like(indices_out)
            indices_tmp[sorted_indices] = indices_out  # invert the sorting
            indices_out = indices_tmp
            distances_tmp = cp.empty_like(distances_out)
            distances_tmp[sorted_indices] = distances_out
            distances_out = distances_tmp
            n_neighbors_tmp = cp.empty_like(n_neighbors_out)
            n_neighbors_tmp[sorted_indices] = n_neighbors_out
            n_neighbors_out = n_neighbors_tmp

        return indices_out, distances_out, n_neighbors_out

    def tree_data(self):
        data = cp.ndarray((self.num_nodes,), dtype=self.tree_dtype, memptr=self.nodes).get()
        return data