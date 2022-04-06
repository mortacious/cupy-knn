from cupy_knn import LBVHIndex
from cupyx.profiler import benchmark
from plyfile import PlyData
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Benchmark knn queries on a dataset')
parser.add_argument("input_file", type=str, help='An input ply-file used for the benchmark')
parser.add_argument("-k", "--knn", type=int, default=16,
                    help='The number of nearest neighbors to find for each point in the input file')
parser.add_argument("-r", "--radius", type=float, default=None,
                    help='The maximum radius the search the nearest neighbors in')
parser.add_argument("-l", "--leafsize", type=int, default=32, help="The maximum size of one leaf in the search tree")

args = parser.parse_args()
pc = PlyData.read(args.input_file)

points = np.stack([pc.elements[0].data['x'], pc.elements[0].data['y'], pc.elements[0].data['z']], axis=1)

points = points[:100]
print(f"Benchmarking KNN Search for {points.shape[0]} points with leafsize {args.leafsize}")
lbvh = LBVHIndex(leaf_size=3)
print(f"Run times for {points.shape[0]} queries with k={args.knn} and r={args.radius}: ")

# build the index
lbvh.build(points)

t = lbvh.tree_data()
for i, d in enumerate(t):
    if d["atomic"] > 0:
        print(i, d)
    else:
        print(i, "---")
