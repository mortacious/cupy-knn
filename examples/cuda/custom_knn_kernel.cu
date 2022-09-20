#include "query_knn.cuh"

using namespace lbvh;

__global__ void custom_knn_kernel(const BVHNode *nodes,
                                 const float3* __restrict__ points,
                                 const unsigned int* __restrict__ sorted_indices,
                                 unsigned int root_index,
                                 const float max_radius,
                                 const float3* __restrict__ query_points,
                                 const unsigned int* __restrict__ sorted_queries,
                                 unsigned int N,
                                 // custom argments
                                 float3* means_out)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;
    unsigned int query_idx = sorted_queries[idx];

    auto queue = query_knn(nodes, points, sorted_indices, root_index, &query_points[query_idx], max_radius);

    // compute the mean of all nearest neighbors found as an example
    float3 sum = make_float3(0.0, 0.0, 0.0);
    for(int i=0; i<queue.size(); ++i) {
        sum += points[queue[i].id];
    }

    // write the result back in an unsorted order
    means_out[query_idx] = sum / queue.size();
}