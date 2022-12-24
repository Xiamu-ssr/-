#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <cuda.h>
#include "common.h"

#define INDENT "    "
#define MAX_THREADS 40960
#define THREADS_PER_VECTOR 4
#define MAX_THREADS_PER_BLOCK 256
#define VECTORS_PER_BLOCK (MAX_THREADS_PER_BLOCK/THREADS_PER_VECTOR)
#define BLOCKS_PER_GRID (MAX_THREADS/MAX_THREADS_PER_BLOCK)
const char* version_name = "naive base-line";\

void preprocess(dist_matrix_t *mat, data_t *x, data_t *y) {
}

void destroy_additional_info(void *additional_info) {
}


template <unsigned int WarpSize>
__device__ __forceinline__ float warpReduceSum(float sum) {
    if (WarpSize >= 32)sum += __shfl_down_sync(0xffffffff, sum, 16); // 0-16, 1-17, 2-18, etc.
    if (WarpSize >= 16)sum += __shfl_down_sync(0xffffffff, sum, 8);// 0-8, 1-9, 2-10, etc.
    if (WarpSize >= 8)sum += __shfl_down_sync(0xffffffff, sum, 4);// 0-4, 1-5, 2-6, etc.
    if (WarpSize >= 4)sum += __shfl_down_sync(0xffffffff, sum, 2);// 0-2, 1-3, 4-6, 5-7, etc.
    if (WarpSize >= 2)sum += __shfl_down_sync(0xffffffff, sum, 1);// 0-1, 2-3, 4-5, etc.
    return sum;
}

template <typename IndexType, typename ValueType>
__global__ void My_spmv_csr_kernel(const IndexType row_num,
                       const IndexType * A_row_offset,
                       const IndexType * A_col_index,
                       const ValueType * A_value,
                       const ValueType * x,
                       ValueType * y)
{
    const IndexType THREADS_PER_BLOCK = VECTORS_PER_BLOCK * THREADS_PER_VECTOR;
    const IndexType thread_id   = THREADS_PER_BLOCK * blockIdx.x + threadIdx.x;    // global thread index
    const IndexType thread_lane = threadIdx.x & (THREADS_PER_VECTOR - 1);          // thread index within the vector
    const IndexType row_id   = thread_id   /  THREADS_PER_VECTOR;               // global vector index

    if(row_id < row_num){
        const IndexType row_start = A_row_offset[row_id];                  //same as: row_start = Ap[row];
        const IndexType row_end   = A_row_offset[row_id+1];

        // initialize local sum
        ValueType sum = 0;

        // accumulate local sums
        for(IndexType jj = row_start + thread_lane; jj < row_end; jj += THREADS_PER_VECTOR)
            sum += A_value[jj] * x[ A_col_index[jj] ];

        sum = warpReduceSum<THREADS_PER_VECTOR>(sum);
        if (thread_lane == 0){
            y[row_id] = sum;
        }   
    }
}

inline int ceiling(int num, int den) {
    return (num - 1) / den + 1;
}

void spmv(dist_matrix_t *mat, const data_t* x, data_t* y) {
    // int *row_counter;
    // cudaMalloc(&row_counter, sizeof(int));
    // cudaMemset(row_counter, 0, sizeof(int));
    int m = mat->global_m;
    // dim3 grid_size (ceiling(m, blockDimx), 1, 1);
    const unsigned int NUM_BLOCKS = static_cast<unsigned int>((m + (VECTORS_PER_BLOCK - 1)) / VECTORS_PER_BLOCK);

    My_spmv_csr_kernel<index_t, data_t><<<NUM_BLOCKS, MAX_THREADS_PER_BLOCK>>>(m, mat->gpu_r_pos, mat->gpu_c_idx, mat->gpu_values, x, y);
    // cudaFree(row_counter);
}
