#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <iostream>
#include <random>
#include <vector>

#define CUDA_CHECK(call)                                                                           \
    do {                                                                                           \
        cudaError_t err = (call);                                                                  \
        if (err != cudaSuccess) {                                                                  \
            fprintf(stderr, "Error: %s , file: %s , line: %d \n", cudaGetErrorString(err),         \
                    __FILE__, __LINE__);                                                           \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    } while (0)

#define TILE 16

// split k sum into phases
// load 16x16 from A and B each phase, each thread grabs one element from gloabl
__global__ void sgemm_tiled(const float* A, const float* B, float* C, int N) {
    __shared__ float tileA[TILE][TILE];
    __shared__ float tileB[TILE][TILE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int j = (TILE * blockIdx.x) + tx; // C[i][j]
    int i = (TILE * blockIdx.y) + ty;

    float c = 0.0f;
    int phases = (N + TILE - 1) / TILE; // round up so any N can work

    for (int p{}; p < phases; p++) {

        int a_col = (p * TILE) + tx;
        int b_row = (p * TILE) + ty;

        tileA[ty][tx] = (i < N && a_col < N) ? A[(i * N) + a_col] : 0.0f;
        tileB[ty][tx] = (b_row < N && j < N) ? B[(b_row * N) + j] : 0.0f;

        __syncthreads(); // both tiles are loaded

        for (int k{}; k < TILE; k++) {
            c += tileA[ty][k] * tileB[k][tx];
        }

        __syncthreads(); // all threads done
    }

    if (i < N && j < N) {
        C[(i * N) + j] = c;
    }
}

void sgemm_cpu(const float* A, const float* B, float* C, int N) {
    // C[i,j] = sum : A[i][0-(N-1)]*B[0-(N-1)][j]
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float c{};
            for (int k = 0; k < N; k++) {
                c += A[(i * N) + k] * B[(k * N) + j];
            }
            C[(i * N) + j] = c;
        }
    }
}

float max_diff(const float* A, const float* B, int N) {
    float max{};
    for (int i = 0; i < N * N; i++) {
        max = std::max(max, std::fabs(A[i] - B[i]));
    }
    return max;
}

int main() {

    for (int N : {256, 512, 1000, 1024}) {
        std::vector<float> A(N * N);
        std::vector<float> B(N * N);
        std::vector<float> C(N * N);
        std::vector<float> D(N * N);

        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (int i = 0; i < N * N; i++) {

            A[i] = dist(rng);
            B[i] = dist(rng);
        }

        // allocate GPU memory
        float* d_A;
        CUDA_CHECK(cudaMalloc(&d_A, (N * N) * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_A, A.data(), N * N * sizeof(float), cudaMemcpyHostToDevice));

        float* d_B;
        CUDA_CHECK(cudaMalloc(&d_B, (N * N) * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_B, B.data(), N * N * sizeof(float), cudaMemcpyHostToDevice));

        float* d_C;
        CUDA_CHECK(cudaMalloc(&d_C, (N * N) * sizeof(float)));

        // decide block size and # of blocks
        dim3 block(TILE, TILE);
        dim3 grid(((N + block.x - 1) / block.x), ((N + block.y - 1) / block.y));

        // warm up
        sgemm_tiled<<<grid, block>>>(d_A, d_B, d_C, N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // time stuff
        float gpu_ms{};

        cudaEvent_t start;
        cudaEvent_t end;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&end));
        CUDA_CHECK(cudaEventRecord(start));

        sgemm_tiled<<<grid, block>>>(d_A, d_B, d_C, N);
        CUDA_CHECK(cudaGetLastError()); // to catch a bad launch

        CUDA_CHECK(cudaEventRecord(end));
        CUDA_CHECK(cudaEventSynchronize(end)); // CPU waits to end to actually happen
        CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, start, end));

        CUDA_CHECK(cudaMemcpy(C.data(), d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_C));

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(end));

        double gpu_seconds = gpu_ms / 1000.0;
        double gpu_billion_ops_per_sec = (2.0 * N * N * N) / (gpu_seconds * 1e9);
        std::cout << "For N=" << N << "\n";
        std::cout << "GPU GFLOP/s = " << gpu_billion_ops_per_sec << "\n";

        sgemm_cpu(A.data(), B.data(), D.data(), N);
        std::cout << "MAX DIFF: " << max_diff(C.data(), D.data(), N) << "\n";
        std::cout << "\n\n\n";
    }
}
