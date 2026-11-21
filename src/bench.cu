// build (from repo root): mkdir -p results && nvcc -O3 -arch=sm_75 -o bench src/bench.cu -lcublas
// && ./bench then: python3 scripts/plot.py

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <fstream>
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

__global__ void sgemm_naive(const float* A, const float* B, float* C, int N) {

    int j = (blockDim.x * blockIdx.x) + threadIdx.x;
    int i = (blockDim.y * blockIdx.y) + threadIdx.y;
    if (i >= N || j >= N) {
        return;
    }
    int idx = (i * N) + j;
    float c = 0.0f;
    for (int k{}; k < N; k++) {
        c += A[(i * N) + k] * B[(k * N) + j];
    }
    C[idx] = c;
}

#define TILE 16

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

        __syncthreads();
    }

    if (i < N && j < N) {
        C[(i * N) + j] = c;
    }
}

#define BM 64      // C rows per block
#define BN 64      // C columns per block
#define BK 16      // width of the k slice loaded per phase
#define TM 4       // C rows in each thread
#define TN 4       // C columns
#define THREADS 16 // block is 16x16 threads

__global__ void sgemm_register_tiled(const float* A, const float* B, float* C, int N) {
    __shared__ float tileA[BM][BK]; // 64x16 slice of A
    __shared__ float tileB[BK][BN]; // 16x64 slice of B

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int t = (ty * THREADS) + tx;     // thread number in block
    int block_row = BM * blockIdx.y; // 1st C row
    int block_col = BN * blockIdx.x;

    float c[TM][TN] = {}; // 16 result
    float a_reg[TM];
    float b_reg[TN];

    int phases = (N + BK - 1) / BK;

    for (int p{}; p < phases; p++) {
        // slice: 64*16 = 1024 floats / 256 threads = 4 f/t
        for (int l{}; l < (BM * BK) / 256; l++) {
            int idx = t + (l * 256);
            int r = idx / BK;
            int col = idx % BK;
            int g_row = block_row + r;
            int g_col = (p * BK) + col;
            tileA[r][col] = (g_row < N && g_col < N) ? A[(g_row * N) + g_col] : 0.0f;
        }
        for (int l{}; l < (BK * BN) / 256; l++) {
            int idx = t + (l * 256);
            int r = idx / BN;
            int col = idx % BN;
            int g_row = (p * BK) + r;
            int g_col = block_col + col;
            tileB[r][col] = (g_row < N && g_col < N) ? B[(g_row * N) + g_col] : 0.0f;
        }

        __syncthreads(); // both slices loaded

        for (int k{}; k < BK; k++) {
            for (int m{}; m < TM; m++) {
                a_reg[m] = tileA[ty + (m * THREADS)][k];
            }
            for (int n{}; n < TN; n++) {
                b_reg[n] = tileB[k][tx + (n * THREADS)];
            }
            for (int m{}; m < TM; m++) {
                for (int n{}; n < TN; n++) {
                    c[m][n] += a_reg[m] * b_reg[n];
                }
            }
        }

        __syncthreads(); // all done
    }

    for (int m{}; m < TM; m++) {
        int i = block_row + ty + (m * THREADS);
        for (int n{}; n < TN; n++) {
            int j = block_col + tx + (n * THREADS);
            if (i < N && j < N) {
                C[(i * N) + j] = c[m][n];
            }
        }
    }
}

void sgemm_cublas(cublasHandle_t handle, const float* d_A, const float* d_B, float* d_C, int N) {
    const float alpha = 1.0f; // C = alpha * (A * B) + beta * C
    const float beta = 0.0f;
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_B, N, d_A, N,
                             &beta, d_C, N));
}

float max_diff(const float* A, const float* B, int N) {
    float max{};
    for (int i = 0; i < N * N; i++) {
        max = std::max(max, std::fabs(A[i] - B[i]));
    }
    return max;
}

using Kernel = void (*)(const float*, const float*, float*, int);

// average ms of one launch, over `runs` launches (after one warm up)
float time_kernel(Kernel kernel, dim3 grid, dim3 block, const float* d_A, const float* d_B,
                  float* d_C, int N, int runs) {
    kernel<<<grid, block>>>(d_A, d_B, d_C, N); // warm up
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    float gpu_ms{};
    cudaEvent_t start;
    cudaEvent_t end;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&end));
    CUDA_CHECK(cudaEventRecord(start));
    for (int r{}; r < runs; r++) {
        kernel<<<grid, block>>>(d_A, d_B, d_C, N);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(end));
    CUDA_CHECK(cudaEventSynchronize(end));
    CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, start, end));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(end));
    return gpu_ms / runs;
}

// same thing for cuBLAS (it's a library call, not a kernel launch)
float time_cublas(cublasHandle_t handle, const float* d_A, const float* d_B, float* d_C, int N,
                  int runs) {
    sgemm_cublas(handle, d_A, d_B, d_C, N); // warm up
    CUDA_CHECK(cudaDeviceSynchronize());

    float gpu_ms{};
    cudaEvent_t start;
    cudaEvent_t end;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&end));
    CUDA_CHECK(cudaEventRecord(start));
    for (int r{}; r < runs; r++) {
        sgemm_cublas(handle, d_A, d_B, d_C, N);
    }
    CUDA_CHECK(cudaEventRecord(end));
    CUDA_CHECK(cudaEventSynchronize(end));
    CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, start, end));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(end));
    return gpu_ms / runs;
}

int main() {
    const int runs = 10;

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH)); // plain FP32, no TF32

    std::ofstream csv("results/bench.csv");
    if (!csv) {
        fprintf(stderr, "can't open results/bench.csv (run from repo root, mkdir results)\n");
        return 1;
    }
    csv << "N,version,ms,gflops,pct_cublas,max_diff\n";

    for (int N : {256, 512, 1024, 2048, 4096}) {
        std::vector<float> A(N * N);
        std::vector<float> B(N * N);
        std::vector<float> C(N * N);
        std::vector<float> REF(N * N); // cuBLAS result, used as the reference

        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (int i = 0; i < N * N; i++) {

            A[i] = dist(rng);
            B[i] = dist(rng);
        }

        float* d_A;
        CUDA_CHECK(cudaMalloc(&d_A, (N * N) * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_A, A.data(), N * N * sizeof(float), cudaMemcpyHostToDevice));

        float* d_B;
        CUDA_CHECK(cudaMalloc(&d_B, (N * N) * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_B, B.data(), N * N * sizeof(float), cudaMemcpyHostToDevice));

        float* d_C;
        CUDA_CHECK(cudaMalloc(&d_C, (N * N) * sizeof(float)));

        // cuBLAS first: its time is 100% and its result is the reference
        // (CPU check is too slow at 4096)
        float cublas_ms = time_cublas(handle, d_A, d_B, d_C, N, runs);
        CUDA_CHECK(cudaMemcpy(REF.data(), d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost));
        double cublas_gflops = (2.0 * N * N * N) / ((cublas_ms / 1000.0) * 1e9);

        dim3 block16(16, 16);
        dim3 grid16(((N + 15) / 16), ((N + 15) / 16));
        dim3 grid64(((N + BN - 1) / BN), ((N + BM - 1) / BM));

        const char* names[] = {"naive", "tiled", "register_tiled"};
        Kernel kernels[] = {sgemm_naive, sgemm_tiled, sgemm_register_tiled};
        dim3 grids[] = {grid16, grid16, grid64};

        std::cout << "For N=" << N << "\n";
        for (int v{}; v < 3; v++) {
            CUDA_CHECK(
                cudaMemset(d_C, 0, N * N * sizeof(float))); // so a kernel that writes nothing fails
            float ms = time_kernel(kernels[v], grids[v], block16, d_A, d_B, d_C, N, runs);
            CUDA_CHECK(cudaMemcpy(C.data(), d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost));

            double gflops = (2.0 * N * N * N) / ((ms / 1000.0) * 1e9);
            double pct = (100.0 * cublas_ms) / ms; // same work, so time ratio = speed ratio
            float diff = max_diff(C.data(), REF.data(), N);

            csv << N << "," << names[v] << "," << ms << "," << gflops << "," << pct << "," << diff
                << "\n";
            std::cout << names[v] << ": " << gflops << " GFLOP/s, " << pct
                      << "% of cuBLAS, MAX DIFF: " << diff << "\n";
        }
        csv << N << ",cublas," << cublas_ms << "," << cublas_gflops << ",100,0\n";
        std::cout << "cublas: " << cublas_gflops << " GFLOP/s\n";
        std::cout << "\n\n\n";

        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_C));
    }

    CUBLAS_CHECK(cublasDestroy(handle));
    std::cout << "wrote results/bench.csv\n";
}
