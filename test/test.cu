#include <cstdio>

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
    cudaDeviceSynchronize();
    return 0;
}
