import torch

import add_one_ext

tensor = torch.ones(10, dtype=torch.float32, device="cuda:0")
print("Before:", tensor)

add_one_ext.add_one(tensor)

print("After:", tensor)
