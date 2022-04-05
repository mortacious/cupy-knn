#define HASH_64 1 // use 64 bit morton codes

#include "aabb.cuh"
#include "lbvh.cuh"

using namespace lbvh;

__global__ void compute_morton_kernel(AABB* __restrict__ const aabbs,
                                      AABB* __restrict__ const extent,
                                      unsigned long long int* morton_codes,
                                      unsigned int N) {
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx >= N)
        return;

    const AABB& box = aabbs[idx];
    morton_codes[idx] = morton_code(box, *extent);
}

__global__ void compute_morton_points_kernel(float3* __restrict__ const points,
                                             AABB* __restrict__ const extent,
                                             unsigned long long int* morton_codes,
                                             unsigned int N) {
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx >= N)
        return;

    const float3& point = points[idx];
    morton_codes[idx] = morton_code(point, *extent);
}

__forceinline__ __device__ void initialize_leaf_node(unsigned int leaf_idx, BVHNode *nodes, const AABB *sorted_aabbs) {
    // Reset leaf nodes
    BVHNode* leaf = &nodes[leaf_idx];

    leaf->bounds = sorted_aabbs[leaf_idx];
    leaf->atomic = 1; // leaf nodes will be processed by the first thread
    leaf->range_right = leaf_idx;
    leaf->range_left = leaf_idx;
    leaf->parent = UINT_MAX;
    leaf->child_left = UINT_MAX;
    leaf->child_right = UINT_MAX;
}

__forceinline__ __device__ void initialize_internal_node(unsigned int internal_index, BVHNode *nodes) {


    auto* internal = &nodes[internal_index];
    internal->atomic = 0; // internal nodes will be processed by the second thread encountering them
    internal->parent = UINT_MAX;
    internal->child_left = UINT_MAX;
    internal->child_right = UINT_MAX;
}

__global__ void initialize_tree_kernel(BVHNode *nodes,
                                       const AABB *sorted_aabbs,
                                       unsigned int N)
{
    unsigned int leaf_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (leaf_idx >= N)
        return;

    // Reset leaf nodes
    initialize_leaf_node(leaf_idx, nodes, sorted_aabbs);

    // Reset internal nodes
    if(leaf_idx < N-1) {
        // Reset internal nodes
        unsigned int internal_index = N + leaf_idx;
        initialize_internal_node(internal_index, nodes);
    }
}

__global__ void construct_tree_kernel(BVHNode *nodes,
                                      unsigned int* root_index,
                                      const unsigned long long int *sorted_morton_codes,
                                      unsigned int N)
{
    unsigned int leaf_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (leaf_idx >= N)
        return;

    // Special case
    if (N == 1)
    {
        BVHNode* leaf = &nodes[leaf_idx];
        nodes[N].bounds = leaf->bounds;
        nodes[N].child_left = leaf_idx;
        root_index[0] = N;
    } else {

        // recurse up to the root building up the tree
        process_parent(leaf_idx, nodes, sorted_morton_codes, root_index, N);
    }
}

__global__ void optimize_tree_kernel(BVHNode *nodes,
                                     unsigned int* root_index,
                                     unsigned int max_node_size,
                                     unsigned int N)
{
    unsigned int leaf_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (leaf_idx >= N)
        return;

    BVHNode* leaf = &nodes[leaf_idx];

    unsigned int current_idx = leaf->parent;
    BVHNode* current = &nodes[current_idx];

    BVHNode* parent;
    unsigned int node_size;

    while(true) {
        if(current_idx == UINT_MAX) {
            //arrived at root by merging all nodes.
            root_index[0] = leaf_idx;
            return; // we are at the root
        }
        const unsigned int parent_idx = current->parent;
        parent = &nodes[parent_idx]; // this might change due to merges

        node_size = current->range_right - current->range_left + 1;

        if(node_size <= max_node_size && leaf_idx <= (current_idx-N)) {
            // only one thread will do this
            make_leaf(current_idx, leaf_idx, nodes, N);
            current->atomic = -1; // mark the current node as invalid to make it removable by the optimization
            current = parent;
            current_idx = parent_idx;
        } else if(node_size <= max_node_size && leaf_idx > current_idx) {
            // the other thread will just set it's leaf to invalid
            // as it will be merged by the other thread and abort
            leaf->atomic = -1;
            return;
        } else {
            return; // nothing to do here so abort
        }
    }
}