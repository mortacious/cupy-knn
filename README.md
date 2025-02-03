# Cupy-KNN

This package provides a fast nearest neighbor index for 3D points 
using a `cupy` implementation of a linear BVH-tree.

## Installation

Since `cupy` can be installed in several different variants (depending on the cuda version; `cupy`, `cupy-cuda11x`, `cupy-cuda12x`), 
`cupy-knn` requires the selection of an optional dependency depending on the installed `cupy` version:

### Using pip
```
pip install cupy-knn[default]
```
```
pip install cupy-knn[cuda11x]
```
```
pip install cupy-knn[cuda12x]
```

## Usage

`cupy-knn` provides the class `LBVHIndex` that wraps an array-based linear bounding volume hierarchy (LBVH) for 3D-points on the gpu accessible using `cupy`.
The LBVH can be created by using the following code:

```python
from cupy_knn import LBVHIndex

lbvh = LBVHIndex(leaf_size=32,
                 compact=True,
                 shrink_to_fit=True,
                 sort_queries=True)
```

Afterwards, it is required to actually build the tree using a provided `cupy`-array of 3D points:

```python
lbvh.build(self, points)
```

This will first sort the `points` array in morton order and construct and optionally compress the tree using multiple CUDA-kernels.
The `LBVHIndex` avoids copiing the `points` array if possible
to conserve space on the GPU so whenever the data in the `points` array changes, the index has to be rebuilt using `build` again.

Afterwards, the `LBVHIndex` can be prepared for KNN as follows:

```python
lbvh.prepare_knn_default(k, radius=np.inf) # use radius=<float> to specifiy a maximum search radius for the neighbors
```

The CUDA implementation relies on the KNN-queue being present in the GPU registers avoiding excessive access to global memory.
Therefore the KNN-query kernel has to be recompiled whenever the k parameter changes. After preparation the query can be executed using:

```python
indices, distances, count = lbvh.query_knn(query_points)
```

Additionally, the `LBVHIndex` supports the injection of custom kernels to process the query results directly. This is required for radius-queries:

```python
lbvh.prepare_radius(custom_module, "custom_radius_kernel", radius=0.5)
```

```python
lbvh.prepare_knn(custom_module, "custom_knn_kernel", radius=0.5)
```

For KNN-queries, the K parameter must be provided to the custom module via compiler flags. These can be obtained by calling `lbvh.compile_flags(k=k)`.

Refer to the examples directory for a more detailed example of custom kernels.


## Acknowledgements

This package is inspired by the approach presented in the following paper:

```bib
@inproceedings{jakob2021optimizing,
  title={Optimizing LBVH-Construction and Hierarchy-Traversal to accelerate kNN Queries on Point Clouds using the GPU},
  author={Jakob, Johannes and Guthe, Michael},
  booktitle={Computer Graphics Forum},
  volume={40},
  number={1},
  pages={124--137},
  year={2021},
  organization={Wiley Online Library}
}
```

