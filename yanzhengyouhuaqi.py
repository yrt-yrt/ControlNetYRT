import torch

# 初始参数
control = torch.tensor([17.3712], requires_grad=True)

# 假设梯度值
grad_value = torch.tensor([0.0154])

# Adam 优化器
optimizer = torch.optim.Adam([control], lr=0.79, betas=(0.9, 0.999), eps=1e-8)

# 手动设置梯度
control.grad = grad_value

# 更新前的参数值
print(f'Before update: {control.item()}')

# 执行一次优化步骤
optimizer.step()

# 更新后的参数值
print(f'After update: {control.item()}')



   

