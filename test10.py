import torch
import torch.nn.functional as F
import numpy as np

# 假设 model 是你的 UNet 模型
class DummyModel(torch.nn.Module):
    def forward(self, x, t, c):
        return x * c  # 假设一个简单的操作

model = DummyModel()
model.eval()

# 示例输入
input_image = torch.randn(1, 3, 64, 64, requires_grad=True)
cond = torch.tensor([[1.0]], requires_grad=True).float()
t = torch.tensor([10])

# p_sample_ddim 函数
def p_sample_ddim(model, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                  temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                  unconditional_guidance_scale=1., unconditional_conditioning=None,
                  dynamic_threshold=None):
    b, *_, device = *x.shape, x.device

    if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
        model_output = model(x, t, c)
    else:
        model_t = model(x, t, c)
        model_uncond = model(x, t, unconditional_conditioning)
        model_output = model_uncond + unconditional_guidance_scale * (model_t - model_uncond)

    e_t = model_output

    # 计算参数
    alphas = np.linspace(0.1, 0.2, 10)  # 假设一些值
    alphas_prev = np.linspace(0.09, 0.19, 10)
    sqrt_one_minus_alphas = np.sqrt(1 - alphas)
    sigmas = np.linspace(0.01, 0.02, 10)

    a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
    a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
    sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
    sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index], device=device)

    # 预测 x_0
    pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
    print(pred_x0.grad)
    # 计算 x_prev
    dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
    noise = sigma_t * torch.randn_like(x) * temperature
    x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
    print(x_prev.grad)
    return x_prev, pred_x0

# 使用优化器来更新 cond
optimizer = torch.optim.Adam([cond], lr=0.01)

# 调用 p_sample_ddim 函数并进行反向传播
for _ in range(10):  # 进行多次优化步骤
    optimizer.zero_grad()

    x_prev, pred_x0 = p_sample_ddim(model, input_image, cond, t, index=5)

    # 计算损失并反向传播
    loss = x_prev.sum()
    loss.backward()

    print(f'Loss: {loss.item()}')
    print(f'Gradient of cond: {cond.grad}')
    print(cond)

    optimizer.step()

print(f'Updated cond: {cond}')

   

