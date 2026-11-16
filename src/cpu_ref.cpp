#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <random>
#include <vector>
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

void sgemm_cpu_reordered(const float* A, const float* B, float* C, int N) {
    // C[i,j] = sum : A[i][0-(N-1)]*B[0-(N-1)][j]
    for (int t = 0; t < N * N; t++) {
        C[t] = 0.0f;
    }
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < N; k++) {
            // float a = A[(i * N) + k]; // fixed and can use in the loop (compiler does this at
            // -O3)
            for (int j = 0; j < N; j++) {
                // c+= A[i][k]*B[k][j];
                C[(i * N) + j] += A[(i * N) + k] * B[(k * N) + j];
            }
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
    float A2[] = {1, 2, 3, 4};
    float B2[] = {5, 6, 7, 8};
    float C2[4];
    sgemm_cpu_reordered(A2, B2, C2, 2);
    printf("%g %g\n%g %g\n\n\n", C2[0], C2[1], C2[2], C2[3]);

    for (int N : {256, 512, 1024}) {
        std::vector<float> A(N * N), B(N * N), C(N * N), D(N * N);

        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (int i = 0; i < N * N; i++) {

            A[i] = dist(rng);
            B[i] = dist(rng);
        }
        auto t0 = std::chrono::steady_clock::now();
        sgemm_cpu(A.data(), B.data(), C.data(), N);
        auto t1 = std::chrono::steady_clock::now();
        double seconds = std::chrono::duration<double>(t1 - t0).count();
        std::cout << "CPU REF:\n";
        std::cout << "Duration for " << N << "x" << N << " is " << seconds << "\n";
        double billion_ops_per_sec = (2.0 * N * N * N) / (seconds * 1e9);
        std::cout << "GFLOP/S: " << billion_ops_per_sec << "\n";
        std::cout << "\n\n\n";

        auto t3 = std::chrono::steady_clock::now();
        sgemm_cpu_reordered(A.data(), B.data(), D.data(), N);
        auto t4 = std::chrono::steady_clock::now();
        double seconds_reordered = std::chrono::duration<double>(t4 - t3).count();
        std::cout << "CPU REF REORDERED:\n";
        std::cout << "Duration for " << N << "x" << N << " is " << seconds_reordered << "\n";
        double billion_ops_per_sec_reordered = (2.0 * N * N * N) / (seconds_reordered * 1e9);
        std::cout << "GFLOP/S: " << billion_ops_per_sec_reordered << "\n";
        std::cout << "\n\n\n";

        std::cout << "MAX DIFF: " << max_diff(C.data(), D.data(), N) << "\n";
        std::cout << "\n\n\n";
    }
    return 0;
}
