//
// Created by mortacious on 3/31/22.
//

#pragma once
#include <cuda/std/cmath>
#include <cuda/std/limits>

#define _min(x, y) (x < y ? x : y)
#define _max(x, y) (x < y ? y : x)
#define CAS(x, y) { auto tmp = _min(x,y); x = _max(x,y); y = tmp; }
#define FLT_MAX 3.402823466e+38F
namespace lbvh {


    template<typename KEY>
    struct KeyType {
        unsigned int id;
        KEY key;

        __device__ KeyType(KEY max_value)
            : id(UINT_MAX), key(max_value) {}

        __device__ KeyType()
            : KeyType(FLT_MAX) {}

        __device__ bool operator <(const KeyType &b) const {
            return key < b.key;
        }
    };

    template<typename KEY, int SIZE>
    struct StaticPriorityQueue {
        int _size;

        KeyType<KEY> _k[SIZE];

        __device__ StaticPriorityQueue(KEY max_value)
            : _size(0) {
            KeyType<KEY> kt(max_value);
            #pragma unroll (SIZE)
            for(int i=0; i<SIZE; ++i) {
                _k[i] = kt;
            }
        }

        __device__ StaticPriorityQueue()
            : StaticPriorityQueue(FLT_MAX) {};

        __device__ KEY top_key() {
            return _k[0].key;
        }

        __device__ int size() {
            return _size;
        }

        __device__ void push(unsigned int index, const KEY& key) {
            KeyType<KEY> k;
            k.key = key;
            k.id = index;
            _k[0] = k;
            _size = min(++_size, SIZE);

            sort();
        }

        __device__ void write_results(unsigned int* indices, KEY* distances, unsigned int* n_neighbors) {
            for(int i=0; i<_size; ++i) {
                KeyType<KEY>& k = _k[(SIZE-i)-1];
                indices[i] = k.id;
                distances[i] = k.key;
            }
            n_neighbors[0] = _size;
        }

        // sort both arrays
        __device__ void sort() {
            #pragma unroll (SIZE-1)
            for (int i = 0; i < (SIZE - 1); ++i) {
                CAS(_k[i], _k[i+1]);
            }
        }
    };
}


#undef CAS
#undef min
#undef max
