from cupy_knn import LBVHIndex
from cupyx.profiler import benchmark
from plyfile import PlyData
import numpy as np
import argparse
import cupy as cp
import pathlib

parser = argparse.ArgumentParser(description='Benchmark knn queries on a dataset')
parser.add_argument("input_file", type=str, help='An input ply-file used for the benchmark')
parser.add_argument("-k", "--knn", type=int, default=16, nargs='+',
                    help='The number of nearest neighbors to find for each point in the input file')
parser.add_argument("-r", "--radius", type=float, default=None,
                    help='The maximum radius the search the nearest neighbors in')

args = parser.parse_args()
pc = PlyData.read(args.input_file)

points = np.stack([pc.elements[0].data['x'], pc.elements[0].data['y'], pc.elements[0].data['z']], axis=1)

lbvh = LBVHIndex(leaf_size=32)

# build the index
lbvh.build(points)

_file_path = pathlib.Path(__file__).parent
with open(_file_path / "cuda/custom_knn_kernel.cu", 'r') as f:
    custom_knn_src = f.read()

for ki in args.knn:
    compile_flags = lbvh.compile_flags(k=ki)
    module = cp.RawModule(code=custom_knn_src,
                          options=compile_flags,
                          name_expressions=('custom_knn_kernel',))

    # prepare the index for knn search with K=16
    lbvh.prepare_knn(module, 'custom_knn_kernel', radius=args.radius)

    means = cp.empty_like(points, dtype=cp.float32)
    # do one query for each of the points in the dataset
    print(benchmark(lbvh.query_knn, (points, means), n_warmup=1, n_repeat=10, name=f"{lbvh.query_knn.__name__}:{ki}"))
    #print(means)
