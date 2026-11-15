#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                                           \
    do {                                                                                           \
        cudaError_t err = (call);                                                                  \
        if (err != cudaSuccess) {                                                                  \
            fprintf(stderr, "Error: %s , file: %s , line: %d \n", cudaGetErrorString(err),         \
                    __FILE__, __LINE__);                                                           \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    } while (0)

__global__ void gprint() {
    printf("Testing!\n");
}

__global__ void print_thread_info() {
    int global = (blockIdx.x * blockDim.x) + threadIdx.x;
    printf("Block: %u   Thread: %u  global: %u \n\n", blockIdx.x, threadIdx.x, global);
}

int main() {
    // //      <<<B, T>>>
    // gprint<<<1, 1>>>();
    print_thread_info<<<2, 4>>>();
    CUDA_CHECK(cudaDeviceSynchronize());
    return 0;
}
