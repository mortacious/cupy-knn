# Cupy-KNN

This package provides a fast nearest neighbor index for 3-D points 
using a Cupy implementation of a left balanced BVH-tree.

## Installation

### Using pip
```
pip install cupy-knn
```

### From source
```
git clone https://github.com/mortacious/cupy-knn.git
cd cupy-knn
python setup.py install
```

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

