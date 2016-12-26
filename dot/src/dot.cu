#include <iostream>
#include <cuda.h>

#define ASYNC 0
#include "dot.h"

int main(int argc, char** argv) {
    size_t pow = read_arg(argc, argv, 1, 4);
    size_t n = (1 << pow);
    auto size_in_bytes = n * sizeof(double);
    int blockNum = read_arg(argc, argv, 2, 4);
    int blockSize = read_arg(argc, argv, 3, 512);

    std::cout << "Single kernel version" << std::endl;
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

    // create events
    CudaEvent start, end;
    start.record();

    // copy to device
    copy_to_device<double>(x_h, x_d, n);
    copy_to_device<double>(y_h, y_d, n);

    // kernel           
    double result = dot_gpu(x_d, y_d, n, blockNum, blockSize);
    end.record();
    end.wait();
    time = end.time_since(start);
    auto expected = dot_host(x_h, y_h, n);
    std::cout << "expected " << expected << " got " << result << std::endl;
    std::cout << "time taken: " << time << std::endl;

    cudaFree(x_d);
    cudaFree(y_d);
    cudaFreeHost(x_h);
    cudaFreeHost(y_h);

    return 0;
}

