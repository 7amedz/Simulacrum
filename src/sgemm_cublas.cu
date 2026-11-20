// build: nvcc -O3 -arch=sm_75 -o sgemm_cublas src/sgemm_cublas.cu -lcublas

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <iostream>
#include <random>
#include <vector>

#include <cublas_v2.h>

#define CUDA_CHECK(call)                                                                           \
    do {                                                                                           \
        cudaError_t err = (call);                                                                  \
        if (err != cudaSuccess) {                                                                  \
            fprintf(stderr, "Error: %s , file: %s , line: %d \n", cudaGetErrorString(err),         \
                    __FILE__, __LINE__);                                                           \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    } while (0)

#define CUBLAS_CHECK(call)                                                                         \
    do {                                                                                           \
        cublasStatus_t st = (call);                                                                \
        if (st != CUBLAS_STATUS_SUCCESS) {                                                         \
            fprintf(stderr, "cuBLAS error: %d , file: %s , line: %d \n", (int)st, __FILE__,        \
                    __LINE__);                                                                     \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    } while (0)

// cuBLAS assumes column-major, our matrices are row-major
// Transpose
// so we ask for C^T = B^T * A^T (column-major) which in memory is row-major C = A * B
// so B is passed before A
void sgemm_cublas(cublasHandle_t handle, const float* d_A, const float* d_B, float* d_C, int N) {
    const float alpha = 1.0f; // C = alpha * (A * B) + beta * C
    const float beta = 0.0f;
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_B, N, d_A, N,
                             &beta, d_C, N));
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
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH)); // plain FP32, no TF32

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

        // warm up: the first launch pays one-time startup costs, don't time it
        sgemm_cublas(handle, d_A, d_B, d_C, N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // time stuff
        float gpu_ms{};

        cudaEvent_t start;
        cudaEvent_t end;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&end));
        CUDA_CHECK(cudaEventRecord(start));

        sgemm_cublas(handle, d_A, d_B, d_C, N);
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

    CUBLAS_CHECK(cublasDestroy(handle));
}
