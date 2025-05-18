#include <cutlass/gemm/device/gemm.h>
#include <iostream>

// Define matrix dimensions
constexpr int M = 128, N = 128, K = 128;

int main() {
    // Define CUTLASS GEMM
    using Gemm = cutlass::gemm::device::Gemm<
        float, cutlass::layout::RowMajor,
        float, cutlass::layout::RowMajor,
        float, cutlass::layout::RowMajor>;

    // Allocate and initialize device memory
    float *A, *B, *C;
    cudaMalloc(&A, M * K * sizeof(float));
    cudaMalloc(&B, K * N * sizeof(float));
    cudaMalloc(&C, M * N * sizeof(float));

    // Run CUTLASS GEMM
    Gemm gemm_op;
    cutlass::gemm::GemmCoord problem_size(M, N, K);
    typename Gemm::Arguments args{problem_size, {A, K}, {B, N}, {C, N}, {C, N}, {1.0f, 0.0f}};
    gemm_op(args);

    cudaDeviceSynchronize();
    std::cout << "CUTLASS GEMM completed.\n";

    cudaFree(A); cudaFree(B); cudaFree(C);
    return 0;
}
