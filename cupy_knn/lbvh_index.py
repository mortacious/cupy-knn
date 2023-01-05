import cupy as cp
import numpy as np
import pathlib
from typing import Optional
from .cuda_util import select_block_grid_sizes, get_cuda_include_path

_cuda_include = get_cuda_include_path()

_file_path = pathlib.Path(__file__).parent

with open(_file_path / "cuda/lbvh_kernels.cu", 'r') as f:
    _lbvh_src = f.read()

_compile_flags = ('--std=c++11', f' -I{pathlib.Path(__file__).parent / "cuda"}',
                  f'-I{_cuda_include}', '--use_fast_math', '--extra-device-vectorization')

_construct_tree_kernels = cp.RawModule(code=_lbvh_src,
                                       options=_compile_flags,
                                       name_expressions=('compute_morton_kernel',
                                                         'compute_morton_points_kernel',
                                                         'initialize_tree_kernel',
                                                         'construct_tree_kernel',
                                                         'optimize_tree_kernel',
                                                         'compute_free_indices_kernel',
                                                         'compact_tree_kernel'))


class LBVHIndex(object):
    """
    A fast left-balanced BVH-tree based nearest neighbor index for 3-D data.

    Parameters
    ----------
    leaf_size: int
        The maximum leaf size in the tree

    sort_queries: bool
        If True the query points will be sorted in morton order prior
        to passing them to the knn query kernel to prevent warp divergence.

    compact: bool
        Compact the tree after optimization by removing the freed space in between. This is done in-place by default.
        If the leaf_size is 1, this parameter is ignored.

    shrink_to_fit: bool
        If the tree is compacted, shrink it's used buffer to fit the new size.
        This requires a full copy, which might not be possible on the gpu depending on the index size.
        If compact is False this parameter is ignored.
    """
    def __init__(self, leaf_size: int = 32, sort_queries: bool = True, compact=True, shrink_to_fit=False):
        self.num_objects = -1
        self.num_nodes = -1
        self.leaf_size = leaf_size
        self.sort_queries = sort_queries
        self.compact = compact
        self.shrink_to_fit = shrink_to_fit

        self.points = None
        self.sorted_indices = None

        self.mode = None
        self.radius = None
        self.extent = None
        self.nodes = None
        self.root_node = None

        self.tree_dtype = cp.dtype({"names": ("aabb", "parent", "child_left",
                                              "child_right", "atomic", "range_left", "range_right"),
                                    "formats": ("6f4", "u4", "u4", "u4", "i4", "u4", "u4")}, align=True)
        self._query_module = None
        self._query_kernel = None

        self._compute_morton_kernel_float = _construct_tree_kernels.get_function('compute_morton_kernel')
        self._compute_morton_points_kernel_float = _construct_tree_kernels.get_function('compute_morton_points_kernel')
        self._construct_tree_kernel_float = _construct_tree_kernels.get_function('construct_tree_kernel')
        self._initialize_tree_kernel_float = _construct_tree_kernels.get_function('initialize_tree_kernel')
        self._optimize_tree_kernel_float = _construct_tree_kernels.get_function('optimize_tree_kernel')
        self._compute_free_indices_kernel = _construct_tree_kernels.get_function('compute_free_indices_kernel')
        self._compact_tree_kernel = _construct_tree_kernels.get_function('compact_tree_kernel')

    @classmethod
    def compile_flags(cls, k=1):
        """
        Obtain the compile flags to be used to compile the custom query modules.

        Parameters
        ----------
        k: int
            The number of nearest neighbors to search in case of a knn module (ignored otherwise).
            This value will be defined in the kernel module as 'K' and is used for the static allocation of the
            search queue.

        Returns
        -------
        compile_flags: tuple
            The compile flags which can be passed directly to the cp.RawModule constructor.

        """
        return _compile_flags + (f'-DK={k}',)

    def prepare_knn_default(self, k: int, radius: Optional[float] = None):
        """
        Prepare the index for KNN search with the specified k and maximum radius.

        Parameters
        ----------
        k: int
            The number of nearest neighbors to search.

        radius: float
            The maximum radius to search the neighbors inside. A value of None will result in an infinite radius.
        """

        with open(_file_path / "cuda/query_knn_kernels.cu", 'r') as f:
            _query_knn_src = f.read()

        module = cp.RawModule(code=_query_knn_src,
                              options=self.compile_flags(k=k),
                              name_expressions=('query_knn_kernel',))
        kernel_name = 'query_knn_kernel'
        self.prepare_knn(module, kernel_name, radius, _custom_knn=k)

    def prepare_knn(self, module: cp.RawModule, kernel_name: str, radius: Optional[float] = None, _custom_knn: int = -1):
        """
        Prepare the index for knn search using the specified custom module and kernel name.

        Parameters
        ----------
        module: cp.RawModule
            The module containing the kernel code for the query. The module should be compiled using the default compiler
            parameters which can be obtained by calling 'compile_flags'. As the number of neighbors has to be static
            for this method, is has to be supplied through the k parameter in 'compile_flags' and will be defined as
            'K' in the module code.

        kernel_name: str
            The name of the query kernel to call. The kernel must support at least the following signature:
                __global__ void query_knn_kernel(
                                 const BVHNode *nodes, // the nodes of the bvh tree
                                 const float3* points, // the points in the bvh tree
                                 const unsigned int* sorted_indices, // the sorted point indices
                                 const unsigned int root_index, // the tree's root index
                                 const float max_radius, // the maximum search radius
                                 const float3* query_points, // the query points
                                 const unsigned int* __restrict__ sorted_queries, // the indices of the query points sorted in morton order if sorting is enabled
                                 const unsigned int N, // the total number of queries
                                 <other custom parameters>
                                 )
            All parameters except the custom ones will be passed to the kernel automatically. any custom parameters must
            be supplied to the 'query_knn' function as additional arguments.

        radius: float
            The maximum search radius.

        _custom_knn: int
            Internal parameter. Not intended for general use.
        """

        self.mode = 'knn'

        if radius is None:
            radius = cp.finfo(cp.float32).max
        else:
            radius = radius**2

        self.radius = radius

        self._custom_knn = _custom_knn
        self._query_module = module
        self._query_kernel = self._query_module.get_function(kernel_name)

    def prepare_radius(self, module: cp.RawModule, kernel_name: str, radius):
        """
        Prepare the index for radius search using the specified custom module and kernel name.

        Parameters
        ----------
        module: cp.RawModule
            The module containing the kernel code for the query.
            The module should be compiled using the default compiler
            parameters which can be obtained by calling 'compile_flags'.

        kernel_name: str
            The name of the query kernel to call. The kernel must support at least the following signature:
            __global__ void query_knn_kernel(
                const BVHNode *nodes, // the nodes of the bvh tree
                const float3* points, // the points in the bvh tree
                const unsigned int* sorted_indices, // the sorted point indices
                const unsigned int root_index, // the tree's root index
                const float max_radius, // the maximum search radius
                const float3* query_points, // the query points
                const unsigned int* __restrict__ sorted_queries, // the indices of the query points sorted in morton order if sorting is enabled
                const unsigned int N, // the total number of queries
                <other custom parameters>
            )
            All parameters except the custom ones will be passed to the kernel automatically. any custom parameters must
            be supplied to the 'query_radius' function as additional arguments.

        radius: float
            The maximum search radius.
        """
        self.mode = 'radius'

        self.radius = radius**2  # squared radius

        self._query_module = module
        self._query_kernel = self._query_module.get_function(kernel_name)

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

                # number of free elements in the optimized tree array
                free_indices_size = int(isum[new_node_count].get())

                free = valid[:free_indices_size]  # reuse the valid space as it is not needed any more

                # compute the free indices
                block_dim, grid_dim = select_block_grid_sizes(isum.device, new_node_count)
                self._compute_free_indices_kernel(grid_dim, block_dim, (valid_sums, isum, free, new_node_count))

                # get the sum of the first object that has to be moved
                first_moved = valid_sums[new_node_count].get()

                block_dim, grid_dim = select_block_grid_sizes(isum.device, self.num_nodes)

                self._compact_tree_kernel(grid_dim,
                                          block_dim,
                                          (self.nodes, root_node, valid_sums_aligned, free, first_moved,
                                           new_node_count, self.num_nodes))

                if self.shrink_to_fit:
                    nodes_old = self.nodes
                    self.nodes = cp.cuda.alloc(self.tree_dtype.itemsize * new_node_count)
                    self.nodes.copy_from_device(nodes_old, self.tree_dtype.itemsize * new_node_count)

                self.num_nodes = new_node_count

        # fetch to root node's location in the tree
        self.root_node = int(root_node.get()[0])

    def _prepare_queries(self, queries: cp.ndarray):
        queries = cp.ascontiguousarray(cp.asarray(queries, dtype=cp.float32)).reshape(-1, 3)

        stream = cp.cuda.get_current_stream()

        # only for large queries: sort them in morton order to prevent too much warp divergence on tree traversal
        if self.sort_queries:
            morton_codes = cp.empty(queries.shape[0], dtype=cp.uint64)
            block_dim, grid_dim = select_block_grid_sizes(queries.device, queries.shape[0])
            self._compute_morton_points_kernel_float(grid_dim, block_dim, (queries, self.extent, morton_codes, queries.shape[0]))
            sorted_indices = cp.argsort(morton_codes).astype(cp.uint32)
            stream.synchronize()
        else:
            sorted_indices = cp.arange(queries.shape[0], dtype=cp.uint32)

        return queries, sorted_indices

    def query_knn(self, queries: cp.ndarray, *args):
        """
        Query the search tree for the nearest neighbors of the query points. 'prepare_knn' must have been
        called prior to colling this function to prepare the cuda kernels for the specified number of neighbors.
        If 'prepare_knn_default' was called instead of 'prepare_knn' with custom query code, the function returns the
        indices, distances and number of neighbors for each query point.
        Otherwise the function returns nothing and all outputs have to be passed through the additional args.

        Parameters
        ----------
        queries: cp.ndarray of shape (n, 3)
            The query points to search the nearest neighbors.

        *args: tuple
            Additional args passed to the kernel. Will be ignored in default mode

        Returns
        -------
        indices: cp.ndarray of shape (n, K)
            The indices of the nearest neighbors.
            Note: This is only returned in default mode (if 'prepare_knn_default' was called)
        distances: cp.ndarray of shape (n, K)
            The squared distances of the nearest neighbors towards their query point
            Note: This is only returned in default mode (if 'prepare_knn_default' was called)
        n_neighbors: cp.ndarray of shape (n,)
            The number of nearest neighbors found
            Note: This is only returned in default mode (if 'prepare_knn_default' was called)
        *args: tuple
            The additional arguments to the function.
            This is returned only if 'prepare_knn' was called directly with custom
            code.

        """
        if self.num_nodes < 0:
            raise ValueError("Index has not been built yet. Call 'build' first.")
        if self.mode != 'knn':
            raise ValueError("Index has not been prepared for knn query. "
                             "Use 'prepare_knn' or 'prepare_knn_default' first.")

        queries, sorted_indices = self._prepare_queries(queries)

        # use the maximum allowed threads per block from the kernel (depends on the number of registers)
        max_threads_per_block = self._query_kernel.attributes['max_threads_per_block']
        block_dim, grid_dim = select_block_grid_sizes(queries.device, queries.shape[0],
                                                      threads_per_block=max_threads_per_block)
        stream = cp.cuda.get_current_stream()

        if self._custom_knn > 0:
            indices_out = cp.full((queries.shape[0], self._custom_knn), cp.iinfo(cp.uint32).max, dtype=cp.uint32)
            distances_out = cp.full((queries.shape[0], self._custom_knn), cp.finfo(cp.float32).max, dtype=cp.float32)
            n_neighbors_out = cp.zeros((queries.shape[0]), dtype=cp.uint32)

            args = (indices_out, distances_out, n_neighbors_out)

        # custom code
        self._query_kernel(grid_dim, block_dim, (self.nodes,
                                                 self.points,
                                                 self.sorted_indices,
                                                 cp.uint32(self.root_node),
                                                 cp.float32(self.radius),
                                                 queries,
                                                 sorted_indices,
                                                 queries.shape[0],
                                                 *args))
        stream.synchronize()

        return args

    def query_radius(self, queries: cp.ndarray, *args):
        """
        Do a radius search on the tree for the query points. 'prepare_radius' must have been
        called prior to colling this function to prepare the cuda kernels for the search. As returning an arbitrary number of
        neighbors for each point is not really feasible from a single cuda kernel,
        this function requires custom kernels to process the neighbors.

        Parameters
        ----------
        queries: cp.ndarray of shape (n, 3)
            The query points to search the nearest neighbors.

        *args: tuple
            Additional args passed to the kernel.

        Returns
        -------
        query_order: cp.ndarray of shape (n,)
            The query indices in morton order. This is returned only if 'prepare_knn' was called directly with custom
            code and sort_queries is True in order to sort the returned values.
        """
        if self.num_nodes < 0:
            raise ValueError("Index has not been built yet. Call 'build' first.")
        if self.mode != 'radius':
            raise ValueError("Index has not been prepared for radius query. Use 'prepare_radius' first.")

        queries, sorted_indices = self._prepare_queries(queries)

        # use the maximum allowed threads per block from the kernel (depends on the number of registers)
        max_threads_per_block = self._query_kernel.attributes['max_threads_per_block']
        block_dim, grid_dim = select_block_grid_sizes(queries.device, queries.shape[0],
                                                      threads_per_block=max_threads_per_block)
        stream = cp.cuda.get_current_stream()

        # custom code
        self._query_kernel(grid_dim, block_dim, (self.nodes,
                                                 self.points,
                                                 self.sorted_indices,
                                                 cp.uint32(self.root_node),
                                                 cp.float32(self.radius),
                                                 queries,
                                                 sorted_indices,
                                                 queries.shape[0],
                                                 *args))
        stream.synchronize()

        return args

    def tree_data(self, numpy: bool = True):
        """
        Returns the tree data containg all nodes as a cupy or numpy structured ndarray.
        As cupy currently does not support structured data types it is recommended to return this as a numpy array.

        Parameters
        ----------
        numpy: bool
            Return as numpy array

        Returns
        -------
        tree_data: ndarray
            The tree's nodes in a structured array

        """
        data = cp.ndarray((self.num_nodes,), dtype=self.tree_dtype, memptr=self.nodes)
        if numpy:
            data = data.get()
        return data
