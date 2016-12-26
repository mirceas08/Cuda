#include <iostream>
#include <cuda.h>

#define ASYNC 1
#include "dot.h"

int main(int argc, char** argv) {
    size_t pow = read_arg(argc, argv, 1, 4);
    size_t n = (1 << pow);
    auto size_in_bytes = n * sizeof(double);
    int blockNum = read_arg(argc, argv, 2, 4);
    int blockSize = read_arg(argc, argv, 3, 512);
    int numChunks = read_arg(argc, argv, 4, 8);
    int chunkSize = n / numChunks;

    std::cout << "Streamed async version" << std::endl;
    std::cout << "dot product CUDA of length n = " << n
              << " : " << size_in_bytes/(1024.*1024.) << "MB"
              << std::endl;

    cuInit(0);

    auto x_h = malloc_host_pinned<double>(n, 2.);
    auto y_h = malloc_host_pinned<double>(n);
    for(auto i=0; i<n; ++i) {
        y_h[i] = rand()%10;
    }

    auto x_d = malloc_device<double>(n);
    auto y_d = malloc_device<double>(n);

    double time; // elapsed time
    double finalResult = 0.0;

    // create events and streams
    CudaEvent start, end;
    cudaStream_t stream[numChunks];
    for (int i = 0; i < numChunks; i++) {
        cudaStreamCreate(&stream[i]);
    }

    // ********** async version ******************
    double result[numChunks];
    start.record(stream[numChunks-1]);
    for (int i = 0; i < numChunks; i++) {
        int offset = i * chunkSize;

        // copy to device
        copy_to_device_async<double>(x_h+offset, x_d+offset, chunkSize, stream[i]);
        copy_to_device_async<double>(y_h+offset, y_d+offset, chunkSize, stream[i]);

        // kernel
        result[i] = dot_gpu(x_d+offset, y_d+offset, chunkSize, blockNum, blockSize, stream[i]);
    }
    end.record(stream[numChunks-1]);
    end.wait();
    for (int i = 0; i < numChunks; i++) {
        finalResult += result[i];
        cudaStreamDestroy(stream[i]);
    }
    time = end.time_since(start);
    auto expected = dot_host(x_h, y_h, n);
    std::cout << "expected " << expected << " got " << finalResult << std::endl;
    std::cout << "time taken: " << time << std::endl;

    cudaFree(x_d);
    cudaFree(y_d);
    cudaFreeHost(x_h);
    cudaFreeHost(y_h);

    return 0;
}

