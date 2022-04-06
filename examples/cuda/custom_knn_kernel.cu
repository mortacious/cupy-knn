#include "query_knn.cuh"

using namespace lbvh;

__global__ void custom_knn_kernel(const BVHNode *nodes,
                                 const float3* points,
                                 const unsigned int* sorted_indices,
                                 unsigned int root_index,
                                 const float max_radius,
                                 const float3* queries,
                                 unsigned int N,
                                 // custom argments
                                 float3* means_out)
{
    unsigned int query_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (query_idx >= N)
        return;

    auto queue = find_KNN(nodes, points, sorted_indices, root_index, &queries[query_idx], max_radius);

    // compute the mean of all nearest neighbors found as an example
    float3 sum = make_float3(0.0, 0.0, 0.0);
    for(int i=0; i<queue.size(); ++i) {
        sum += points[queue[i].id];
    }
    means_out[query_idx] = sum / queue.size();
}