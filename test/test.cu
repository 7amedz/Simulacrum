#include <cstdio>

__global__ void gprint() {
    printf("Testing!\n");
}

int main() {
    gprint<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}
