import torch
import numpy as np

# 示例的 PyTorch Tensor
alphas = torch.tensor([0.9983, 0.9698, 0.9376, 0.9016, 0.8621, 0.8193, 0.7736, 0.7255, 0.6754,
                       0.6241, 0.5720, 0.5199, 0.4684, 0.4181, 0.3697, 0.3236, 0.2804, 0.2403,
                       0.2037, 0.1707, 0.1413, 0.1155, 0.0933, 0.0744, 0.0585, 0.0453, 0.0347,
                       0.0261, 0.0194, 0.0142, 0.0102, 0.0072, 0.0050], device='cuda:0')
print(alphas[5])
# 将 PyTorch Tensor 转换为 NumPy 数组并进行形状调整
alphas_np = alphas.cpu().numpy().reshape(-1)
print(alphas_np[5])
print(alphas_np)


   

