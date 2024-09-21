from share import *
import config

import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random
from torch.optim import LBFGS
from torchviz import make_dot

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler


model_name = 'control_v11f1e_sd15_tile'
model = create_model(f'./models/{model_name}.yaml').cpu()
#model.load_state_dict(load_state_dict('./models/v1-5-pruned.ckpt', location='cuda'), strict=False)
#torch.cuda.empty_cache()
model.load_state_dict(load_state_dict(f'./models/epoch=276-step=27700.ckpt'), strict=False) #有改动1
model = model.cuda()
ddim_sampler = DDIMSampler(model)

#加了一个control_num_str
def process(input_image, control, prompt, a_prompt, n_prompt, num_samples, ddim_steps, guess_mode, strength, scale, seed, eta, denoise_strength):
    global preprocessor

    with torch.no_grad():
        input_image = HWC3(input_image)
        
        img = cv2.resize(input_image, (320,320))
        
        H, W, C = img.shape

        img = torch.from_numpy(img.copy()).float().cuda() / 127.0 - 1.0
        img = img.requires_grad_()
        img = torch.stack([img for _ in range(num_samples)], dim=0)
        img = einops.rearrange(img, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        ddim_sampler.make_schedule(ddim_steps, ddim_eta=eta, verbose=True)
        t_enc = min(int(denoise_strength * ddim_steps), ddim_steps - 1)
        z = model.get_first_stage_encoding(model.encode_first_stage(img))
        z_enc = ddim_sampler.stochastic_encode(z, torch.tensor([t_enc] * num_samples).to(model.device))
        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)
        # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01

        samples = ddim_sampler.decode(z_enc, cond, t_enc, unconditional_guidance_scale=scale, unconditional_conditioning=un_cond)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
    return results

input_image_path = '/home/yuanrt/ControlNet1_1/ControlNet-v1-1-nightly/00013-0-bag.png'
input_image = cv2.imread(input_image_path)
target_image_path = '/home/yuanrt/ControlNet1_1/ControlNet-v1-1-nightly/dataForTrain/resultData/cropped_4_bag.png'
target_image = cv2.imread(target_image_path)

control = torch.tensor([[1]], requires_grad = True, dtype=torch.float32)#.cuda()
target = torch.tensor(target_image, dtype=torch.float32, requires_grad = True).cuda()

prompt = "simple background, white background"
a_prompt = "best quality"
n_prompt = "blur, lowres, bad anatomy, bad hands, cropped, worst quality"
num_samples = 1
ddim_steps = 32
guess_mode = False
strength = 1.0
scale = 9.0
seed = 12345
eta = 1.0
denoise_strength = 1.0

loss_fn = torch.nn.MSELoss()

control = control.requires_grad_()
optimizer = torch.optim.Adam([control], lr=0.0001)

for i in range(10):
    #optimizer.zero_grad()
    control = control.requires_grad_()
    control_s = control.cuda()
    result = process(input_image, control_s, prompt, a_prompt, n_prompt, num_samples, ddim_steps, guess_mode, strength, scale, seed, eta, denoise_strength)
    res = result[0]
    pred = torch.tensor(res, dtype=torch.float32, requires_grad = True).cuda()
    print(target.shape)
    print(pred.shape)
    loss = loss_fn(target, pred)
    print(control.grad)
    control.retain_grad()
    
    
    
    loss.backward()
    
    dot = make_dot(loss, params={'control':control})
    dot.render('control_gradient', format='png')
    optimizer.step()
    print(loss)
    print(control)
   

