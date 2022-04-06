#pragma once
#include "lbvh.cuh"
#include "static_priorityqueue.cuh"
#include "vec_math.h"

// default is one nearest neighbor
#ifndef K
#define K 1
#endif

namespace lbvh {
    __forceinline__ __device__ void push_points_of_node(const BVHNode *node,
                                                        const float3* __restrict__ points,
                                                        const unsigned int* __restrict__ sorted_indices,
                                                        const float3* __restrict__ query_point,
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

    __device__ void find_KNN(const BVHNode* __restrict__ nodes,
                             const float3* __restrict__ points,
                             const unsigned int* __restrict__ sorted_indices,
                             unsigned int root_index,
                             const float3* __restrict__ query_point,
                             StaticPriorityQueue<float, K>& queue)
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
            const float dl = dist_2_aabb(*query_point, child_left->bounds);
            const float dr = dist_2_aabb(*query_point, child_right->bounds);

            if(!bt && is_leaf(child_left) && dl <= queue.top_key()) {
                push_points_of_node(child_left, points, sorted_indices, query_point, queue);
            }
            if(!bt && is_leaf(child_right) && dr <= queue.top_key()) {
                push_points_of_node(child_right, points, sorted_indices, query_point, queue);
            }

            float top = queue.top_key();
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
                float mind(INFINITY);

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
        __syncwarp(); // synchronize the warp before any other operation

    }

    __device__ StaticPriorityQueue<float, K> find_KNN(const BVHNode* __restrict__ nodes,
                             const float3* __restrict__ points,
                             const unsigned int* __restrict__ sorted_indices,
                             unsigned int root_index,
                             const float3* __restrict__ query_point,
                             const float max_radius)
    {

        StaticPriorityQueue<float, K> queue(max_radius);
        find_KNN(nodes, points, sorted_indices, root_index, query_point, queue);
        return queue;
    }
}

