from cupy_knn import LBVHIndex
from cupyx.profiler import benchmark
from plyfile import PlyData
import numpy as np
import argparse
import cupy as cp
import pathlib

parser = argparse.ArgumentParser(description='Benchmark custom radius queries on a dataset')
parser.add_argument("input_file", type=str, help='An input ply-file used for the benchmark')
parser.add_argument("-r", "--radius", type=float, default=0.05,
                    help='The maximum radius the search')

args = parser.parse_args()
pc = PlyData.read(args.input_file)

points = np.stack([pc.elements[0].data['x'], pc.elements[0].data['y'], pc.elements[0].data['z']], axis=1)

lbvh = LBVHIndex(leaf_size=32)

# build the index
lbvh.build(points)

_file_path = pathlib.Path(__file__).parent
with open(_file_path / "cuda/custom_radius_kernel.cu", 'r') as f:
    custom_radius_src = f.read()

compile_flags = lbvh.compile_flags()
module = cp.RawModule(code=custom_radius_src,
                      options=compile_flags,
                      name_expressions=('custom_radius_kernel',))

lbvh.prepare_radius(module, 'custom_radius_kernel', args.radius)

means = cp.empty_like(points, dtype=cp.float32)
n_neighbors = cp.empty(points.shape[0], dtype=cp.uint32)
# do one query for each of the points in the dataset
print(benchmark(lbvh.query_radius, (points, means, n_neighbors), n_warmup=1, n_repeat=10))
print("means", means)
print("number of neighbors", n_neighbors)
print("max neighbors", cp.max(n_neighbors))
