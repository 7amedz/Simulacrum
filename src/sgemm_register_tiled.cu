
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

#define BM 64      // C rows per block
#define BN 64      // C columns per block
#define BK 16      // width of the k slice loaded per phase
#define TM 4       // C rows in each thread
#define TN 4       // C columns
#define THREADS 16 // block is 16x16 threads

// every block does a 64x64 portion of C
// each thread a 4x4 group (16 values in registers)
// each k: 4 values of A and 4 values of B from shared memory
// 8 reads for 16 */+ vs 2/1 in tiled kernel
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
        // 16x16 threads, but each block covers 64x64 of C
        dim3 block(THREADS, THREADS);
        dim3 grid(((N + BN - 1) / BN), ((N + BM - 1) / BM));

        // warm up: the first launch pays one-time startup costs, don't time it
        sgemm_register_tiled<<<grid, block>>>(d_A, d_B, d_C, N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // time stuff
        float gpu_ms{};

        cudaEvent_t start;
        cudaEvent_t end;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&end));
        CUDA_CHECK(cudaEventRecord(start));

        sgemm_register_tiled<<<grid, block>>>(d_A, d_B, d_C, N);
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
