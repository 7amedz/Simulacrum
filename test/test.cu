#include <cstdio>

__global__ void gprint() {
    printf("Testing!\n");
}

int main() {
    gprint<<<1, 1>>>();
    return 0;
}
