#include <cstdio>

__global__ void gprint() {
    printf("Testing!\n");
}

__global__ void print_thread_info() {
    printf("Block: %d   Thread: %d  ", blockId.x, threadId.x);
    int global = (blockId.x * blockDim.x) + threadId.x;
    printf("global: %d \n\n", global);
}

int main() {
    // //      <<<B, T>>>
    // gprint<<<1, 1>>>();
    print_thread_info<<<2, 4>>>();
    cudaDeviceSynchronize();
    return 0;
}
