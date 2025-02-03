from cupy_knn import LBVHIndex
from cupyx.profiler import benchmark
from plyfile import PlyData
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Benchmark knn queries on a dataset')
parser.add_argument("input_file", type=str, help='An input ply-file used for the benchmark')
parser.add_argument("-k", "--knn", type=int, nargs='+', default=[16],
                    help='The number of nearest neighbors to find for each point in the input file')
parser.add_argument("-r", "--radius", type=float, default=None,
                    help='The maximum radius the search the nearest neighbors in')
parser.add_argument("-l", "--leafsize", type=int, default=32, help="The maximum size of one leaf in the search tree")
parser.add_argument("-c", "--compact", action='store_true', help="Compact the tree to remove unused spaces in between")
parser.add_argument("-s", "--shrink_to_fit", action='store_true', help="Shrink to buffer size of the tree to match "
                                                                       "it's content after compaction")
parser.add_argument("--sort", action='store_true', help="Sort the queries by their morton index before execution")

args = parser.parse_args()
pc = PlyData.read(args.input_file)

points = np.stack([pc.elements[0].data['x'], pc.elements[0].data['y'], pc.elements[0].data['z']], axis=1)

print(f"Benchmarking KNN Search for {points.shape[0]} points with leafsize {args.leafsize} and compact={args.compact} + "
      f"shrink to fit={args.shrink_to_fit} + sort={args.sort}.")
lbvh = LBVHIndex(leaf_size=args.leafsize,
                 compact=args.compact,
                 shrink_to_fit=args.shrink_to_fit,
                 sort_queries=args.sort)
print(f"Run times for {points.shape[0]} queries with k={args.knn} and r={args.radius}: ")

# build the index
print(benchmark(lbvh.build, (points,), n_repeat=10, n_warmup=1))

for ki in args.knn:
    # prepare the index for knn search with K=16
    lbvh.prepare_knn_default(ki, radius=args.radius)
    # do one query for each of the points in the dataset
    print(benchmark(lbvh.query_knn, (points,), n_repeat=10, n_warmup=1, name=f"{lbvh.query_knn.__name__}:{ki}"))
