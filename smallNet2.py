import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchviz import make_dot

# 定义一个简单的图像处理网络
class ImageNet(nn.Module):
    def __init__(self):
        super(ImageNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 3, 3, padding=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.sigmoid(self.conv3(x))
        return x

# 定义一个处理数值数据的网络
class NumberNet(nn.Module):
    def __init__(self):
        super(NumberNet, self).__init__()
        self.fc1 = nn.Linear(1, 10)
        self.fc2 = nn.Linear(10, 3*320*320)  # 假设图像分辨率为 320x320
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = x.view(-1, 3, 320, 320)
        return x

# 定义一个组合网络
class CombinedNet(nn.Module):
    def __init__(self):
        super(CombinedNet, self).__init__()
        self.image_net = ImageNet()
        self.number_net = NumberNet()
        
    def forward(self, img, num):
        img_feat = self.image_net(img)
        num_feat = self.number_net(num)
        combined_feat = img_feat + num_feat
        return combined_feat

# 初始化网络
model = CombinedNet()#.cuda()

# 损失函数和优化器


# 假设输入数据
input_image = torch.randn(1, 3, 320, 320).type(torch.FloatTensor)#.cuda()  # 输入图像
input_number = torch.tensor([[0.5]]).type(torch.FloatTensor)  # 输入数值
target_image = torch.randn(1, 3, 320, 320).type(torch.FloatTensor)#.cuda()  # 目标图像

input_number = input_number.requires_grad_()
optimizer = torch.optim.Adam([input_number], lr=0.01)
# 前向传播
def p_sample_ddim(model, x, c, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                  temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                  unconditional_guidance_scale=1., unconditional_conditioning=None,
                  dynamic_threshold=None):
    b, *_, device = *x.shape, x.device

    if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
        model_output = model(x, c)
    else:
        model_t = model(x, c)
        #model_uncond = model(x, unconditional_conditioning)
        #model_output = model_uncond + unconditional_guidance_scale * (model_t - model_uncond)

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
    #print(pred_x0.grad)
    # 计算 x_prev
    dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
    noise = sigma_t * torch.randn_like(x) * temperature
    x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
    #print(x_prev.grad)
    return x_prev, pred_x0

# 使用优化器来更新 cond
optimizer = torch.optim.Adam([input_number], lr=0.1)
loss_fun=torch.nn.MSELoss()
# 调用 p_sample_ddim 函数并进行反向传播
for _ in range(20):  # 进行多次优化步骤
    optimizer.zero_grad()
    #input_number_s = input_number.cuda()
    x_prev, pred_x0 = p_sample_ddim(model, input_image, input_number, index=5)

    # 计算损失并反向传播
    #x_prev#.cuda()
    loss = loss_fun(x_prev, target_image)
    loss.backward()
    #dot = make_dot(loss, params={'input_number':input_number})
    #dot.render(f'input_number_gradient_2', format='png')
    print(f'Loss: {loss.item()}')
    print(f'Gradient of cond: {input_number.grad}')
    print(input_number)

    optimizer.step()

#print(f'Updated cond: {input_number}')

"""
loss_fun=torch.nn.MSELoss()
for i in range(10):
    input_number_s = input_number.cuda()
    output_image = net(input_image, input_number_s).cuda()
    loss=loss_fun(output_image, target_image)
    print(input_number.grad)
    #input_number.retain_grad()
    loss.backward()
    #input_number=input_number-input_number.grad*0.01
    dot = make_dot(loss, params={'input_number':input_number})
    dot.render(f'input_number_gradient_{i}', format='png')
    optimizer.step()
    print('the loss is:',loss)
    print(input_number)
"""
