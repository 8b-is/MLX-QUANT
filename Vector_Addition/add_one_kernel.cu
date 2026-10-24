#include <torch/extension.h>

#include <hip/hip_runtime.h>

// GPU kernel, one thread per element
__global__ void add_one(float* data, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n)
    data[idx] += 1.0f;
}

// Launcher, bridges torch::Tensor to raw pointer, sets grid/block, runs kernel
void add_one_launcher(torch::Tensor tensor) {
  int n = tensor.numel();
  float* data = tensor.data_ptr<float>();

  int block_size = 256;
  int grid_size = (n + block_size - 1) / block_size;

  add_one<<<grid_size, block_size>>>(data, n);
  hipDeviceSynchronize();
}

// Python binding, exposes add_one_launcher as add_one_ext.add_one
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("add_one", &add_one_launcher, "Add one kernel (HIP)");
}
