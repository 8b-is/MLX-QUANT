#include <torch/extension.h>

#include <hip/hip_runtime.h>

#define BLOCK 16

// GPU kernel, one thread per output element of C
__global__ void matmul(float* A, float* B, float* C, int M, int N, int K) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < M && col < K) {
    float sum = 0.0f;
    for (int n = 0; n < N; n++) {
      sum += A[row * N + n] * B[n * K + col];
    }
    C[row * K + col] = sum;
  }
}

// Launcher, extracts dims from torch::Tensor, allocates C, sets 2D grid/block
torch::Tensor matmul_launcher(torch::Tensor A, torch::Tensor B) {
  int M = A.size(0), N = A.size(1), K = B.size(1);
  auto C = torch::zeros({M, K}, A.options());

  dim3 block(BLOCK, BLOCK);
  dim3 grid((K + BLOCK - 1) / BLOCK, (M + BLOCK - 1) / BLOCK);

  matmul<<<grid, block>>>(
      A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);
  hipDeviceSynchronize();
  return C;
}

// Python binding, exposes matmul_launcher as matmul_ext.matmul
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "matmul",
      &matmul_launcher,
      "Naive matmul kernel (HIP): A(M,N) @ B(N,K) -> C(M,K)");
}
