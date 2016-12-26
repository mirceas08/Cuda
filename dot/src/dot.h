    #pragma once

#include "util.h"
#include "CudaEvent.h"
#include "CudaStream.h"

// host implementation of dot product
double dot_host(const double *x, const double* y, int n) {
    double sum = 0;
    for(auto i=0; i<n; ++i) {
        sum += x[i]*y[i];
    }
    return sum;
}

// device implementation of dot product
// returns an array of gridDim size
__global__
void dot_gpu_kernel(const double *x, const double* y, double *result, int n)
{
    extern __shared__ double cache[];
    int tId = threadIdx.x + blockIdx.x * blockDim.x;
    int ltId = threadIdx.x;
    cache[ltId] = 0.0;

    double tempCache = 0.0;
    while (tId < n) {
        tempCache += x[tId] * y[tId];
        tId += blockDim.x * gridDim.x;
    }
    cache[ltId] = tempCache;
    __syncthreads();

    int temp = blockDim.x;
    int i = blockDim.x / 2;
    if (ltId < i) {
        while (i != 0) {
            if (temp % 2 == 0) {
                if (ltId < i) {
                    cache[ltId] += cache[ltId + i]; 
                }    
            }
            else {
                if (ltId < i) {
                    cache[ltId] += cache[ltId + i]; 
                }
                if (ltId == 0) {
                    cache[ltId] += cache[2*i];
                }    
            }
            
            __syncthreads();
            temp = i;
            i /= 2; 
        }
    }

    if (ltId == 0)
        result[blockIdx.x] = cache[0];
}

// accumulate the result of each block into a single value
double dot_gpu(const double *x, const double* y, int n, int blockNum, int blockSize, cudaStream_t stream=NULL) {
    double* result = malloc_device<double>(blockNum);
    dot_gpu_kernel<<<blockNum, blockSize, blockSize * sizeof(double), stream>>>(x, y, result, n);

    #if ASYNC
        double* r = malloc_host_pinned<double>(blockNum);
        copy_to_host_async<double>(result, r, blockNum, stream);

        CudaEvent copyResult;
        copyResult.record(stream);
        copyResult.wait();
    #else
        double* r = malloc_host<double>(blockNum);
        copy_to_host<double>(result, r, blockNum);
    #endif

    double dotProduct = 0.0;
    for (int i = 0; i < blockNum; i ++) {
        dotProduct += r[i];
    }

    cudaFree(result);
    #if ASYNC
        cudaFreeHost(r);
    #else
        free(r);
    #endif

    return dotProduct;
}
