#include "query.cuh"

using namespace lbvh;

namespace {
    // a custom handler for a radius query
    struct RadiusHandler {
        float3 point_sum;
        unsigned int cnt;
        float max_radius;

        __device__ RadiusHandler(float radius_sq)
                : point_sum(make_float3(0.0, 0.0, 0.0)), cnt(0), max_radius(radius_sq) {}

        // required functions
        __device__ unsigned int size() const {
            return 0; // always return 0 to process all points
        };

        __device__ unsigned int max_size() const {
            return UINT_MAX; // process all points
        };

        __device__ float max_distance() const {
            return max_radius;
        }

        __device__ void operator()(const float3& point, unsigned int index, float dist) {
            point_sum += point;
            cnt++;
        }

        __device__ float3 mean() const {
            return point_sum / cnt;
        }

        __device__ unsigned int count() const {
            return cnt;
        }
    };
}

__global__ void custom_radius_kernel(const BVHNode *nodes,
                                  const float3* __restrict__ points,
                                  const unsigned int* __restrict__ sorted_indices,
                                  unsigned int root_index,
                                  const float max_radius,
                                  const float3* __restrict__ query_points,
                                  const unsigned int* __restrict__ sorted_queries,
                                  unsigned int N,
                                  // custom argments
                                  float3* means_out,
                                  unsigned int* num_neighbors_out)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
        return;
    unsigned int query_idx = sorted_queries[idx];

    RadiusHandler handler(max_radius);
    query(nodes, points, sorted_indices, root_index, &query_points[query_idx], handler);
    means_out[query_idx] = handler.mean();
    num_neighbors_out[query_idx] = handler.count();
}