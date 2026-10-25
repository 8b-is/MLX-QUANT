import torch

import matmul_ext

A = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32, device="cuda:0")
B = torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32, device="cuda:0")

print("A:", A)
print("B:", B)

C = matmul_ext.matmul(A, B)

print("Result:", C)
