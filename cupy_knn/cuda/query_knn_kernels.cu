//
// Created by mortacious on 3/31/22.
//

#include "lbvh.cuh"
#include "static_priorityqueue.cuh"
#include "vec_math.h"

// default is one nearest neighbor
#ifndef K
    #define K 1
#endif

using namespace lbvh;

__forceinline__ __device__ void push_points_of_node(const BVHNode *node,
                                                    const float3* points,
                                                    const unsigned int* sorted_indices,
                                                    const float3* query_point,
                                                    StaticPriorityQueue<float, K>& queue)
{
    for(int i=node->range_left; i<=node->range_right; ++i) { // range is inclusive!
        auto index = sorted_indices[i];
        auto point = points[index];
        float dist = sq_length3(point-*query_point);

        if(dist <= queue.top_key()) {
            queue.push(index, dist);
        }

    }
}

template<typename T>
__device__ void find_KNN(const BVHNode *nodes,
                         const float3* points,
                         const unsigned int* sorted_indices,
                         unsigned int root_index,
                         const float3* query_point,
                         StaticPriorityQueue<T, K>& queue)
{
    bool bt = false;
    unsigned int last_idx = UINT_MAX;
    unsigned int current_idx = root_index;

    const BVHNode* current = &nodes[current_idx];

    unsigned int parent_idx;

    do {
        parent_idx = current->parent;
        const auto parent = &nodes[parent_idx];

        const unsigned int child_l = current->child_left;
        const unsigned int child_r = current->child_right;

        const auto child_left = &nodes[child_l];
        const auto child_right = &nodes[child_r];
        const T dl = dist_2_aabb(*query_point, child_left->bounds);
        const T dr = dist_2_aabb(*query_point, child_right->bounds);

        if(!bt && is_leaf(child_left) && dl <= queue.top_key()) {
            push_points_of_node(child_left, points, sorted_indices, query_point, queue);
        }
        if(!bt && is_leaf(child_right) && dr <= queue.top_key()) {
            push_points_of_node(child_right, points, sorted_indices, query_point, queue);
        }

        T top = queue.top_key();
        int hsize = queue.size();

        bool traverse_l = (!is_leaf(child_left) && !(hsize == K && dl > top));
        bool traverse_r = (!is_leaf(child_right) && !(hsize == K && dr > top));

        const unsigned int best_idx = (dl <= dr) ? child_l: child_r;
        const unsigned int other_idx = (dl <= dr) ? child_r: child_l;
        if(!bt) {
            if(!traverse_l && !traverse_r) {
                // we do not traverse, so backtrack in next iteration
                bt = true;
                last_idx = current_idx;
                current_idx = parent_idx;
                current = parent;
            } else {
                last_idx = current_idx;
                current_idx = (traverse_l) ? child_l : child_r;
                if (traverse_l && traverse_r) {
                    current_idx = best_idx; // take the best one if both are true
                }
            }
        } else {
            T mind(INFINITY);

            const auto other = &nodes[other_idx];

            if(!is_leaf(other)) {
                mind = (dl <= dr) ? dr: dl;
            }
            if(!is_leaf(other) && (last_idx == best_idx) && mind <= top) {
                last_idx = current_idx;
                current_idx = other_idx;
                bt = false;
            } else {
                last_idx = current_idx;
                current_idx = current->parent;
            }
        }

        // get the next node
        current = &nodes[current_idx];
    } while(current_idx != UINT_MAX);
}

__global__ void query_knn_kernel(const BVHNode *nodes,
                                 const float3* points,
                                 const unsigned int* sorted_indices,
                                 unsigned int root_index,
                                 const float max_radius,
                                 const float3* queries,
                                 unsigned int* indices_out,
                                 float* distances_out,
                                 unsigned int* n_neighbors_out,
                                 unsigned int N)
{
    unsigned int query_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (query_idx >= N)
        return;
    StaticPriorityQueue<float, K> queue(max_radius);
    find_KNN(nodes, points, sorted_indices, root_index, &queries[query_idx], queue);
    __syncwarp(); // synchronize the warp before the write operation
    queue.write_results(&indices_out[query_idx * K], &distances_out[query_idx * K], &n_neighbors_out[query_idx]); // write back the results
}