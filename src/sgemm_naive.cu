

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <iostream>
#include <random>
#include <vector>

// #include <__clang_cuda_builtin_vars.h>
// #include <__clang_cuda_runtime_wrapper.h>

#define CUDA_CHECK(call)                                                                           \
    do {                                                                                           \
        cudaError_t err = (call);                                                                  \
        if (err != cudaSuccess) {                                                                  \
            fprintf(stderr, "Error: %s , file: %s , line: %d \n", cudaGetErrorString(err),         \
                    __FILE__, __LINE__);                                                           \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    } while (0)

__global__ void sgemm_naive(const float* A, const float* B, float* C, int N) {

    int j = (blockDim.x * blockIdx.x) + threadIdx.x;
    int i = (blockDim.y * blockIdx.y) + threadIdx.y;
    int idx = (i * N) + j;
    float c = 0.0f;
    for (int k{}; k < N; k++) {
        c += A[(i * N) + k] * B[(k * N) + j];
    }
    C[idx] = c;
}

void sgemm_cpu(const float* A, const float* B, float* C, int N) {
    // C[i,j] = sum : A[i][0-(N-1)]*B[0-(N-1)][j]
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float c{};
            for (int k = 0; k < N; k++) {
                // c+= A[i][k]*B[k][j];
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

    for (int N : {256, 512, 1024}) {
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
        dim3 block(16, 16);
        dim3 grid((N / block.x), (N / block.y));
        // dim3 grid((N / 16), (N / 16)); how many blocks across and down

        // time stuff
        float gpu_ms{};

        cudaEvent_t start;
        cudaEvent_t end;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&end));
        CUDA_CHECK(cudaEventRecord(start));

        sgemm_naive<<<grid, block>>>(d_A, d_B, d_C, N);
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
        std::cout << "GPU GFLOP/s = " << gpu_billion_ops_per_sec << "\n\n\n";

        std::cout << "\n\n\n";

        auto t0 = std::chrono::steady_clock::now();
        sgemm_cpu(A.data(), B.data(), D.data(), N);
        auto t1 = std::chrono::steady_clock::now();
        double seconds = std::chrono::duration<double>(t1 - t0).count();
        double billion_ops_per_sec = (2.0 * N * N * N) / (seconds * 1e9);
        std::cout << "CPU GFLOP/s:" << billion_ops_per_sec << "\n\n\n";
        //

        std::cout << "MAX DIFF: " << max_diff(C.data(), D.data(), N) << "\n";
        std::cout << "\n\n\n";
        //
        // auto t3 = std::chrono::steady_clock::now();
        // auto t4 = std::chrono::steady_clock::now();
        // double seconds_reordered = std::chrono::duration<double>(t4 - t3).count();
        // double billion_ops_per_sec_reordered = (2.0 * N * N * N) / (seconds_reordered * 1e9);
        // std::cout << "\n\n\n";
        //
        // std::cout << "\n\n\n";
    }
}
