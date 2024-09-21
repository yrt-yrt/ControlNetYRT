from share import *
import config

import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random

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
        #detected_map = input_image.copy()
        img = cv2.resize(input_image, (320,320))
        #img = resize_image(input_image, image_resolution)
        H, W, C = img.shape

        #detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)

        #control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        #control = torch.stack([control for _ in range(num_samples)], dim=0)
        #control = einops.rearrange(control, 'b h w c -> b c h w').clone()
        #control_num = [int(char) for char in control_num_str]
        #print(control_num)
        #control = torch.tensor([control_num])
        #print(control)
        #print(control.shape)
        control = control.to('cuda:0')
        control = control.to(torch.float32)
        #print(control)
        #print(control.shape)

        img = torch.from_numpy(img.copy()).float().cuda() / 127.0 - 1.0
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
        #print(z_enc)

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
        saved_paths = []
        for i, result in enumerate(results):
            save_path = f'1dian5_result_bag.jpg'  # 可以根据需要修改文件名和格式
            cv2.imwrite(save_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))  # 注意颜色通道顺序可能需要转换
            saved_paths.append(save_path)
    return saved_paths

input_image_path = '/home/yuanrt/ControlNet1_1/ControlNet-v1-1-nightly/00013-0-bag.png'
input_image = cv2.imread(input_image_path)

control_num = 1.5
control = torch.tensor([control_num])
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

saved_paths = process(input_image, control, prompt, a_prompt, n_prompt, num_samples, ddim_steps, guess_mode, strength, scale, seed, eta, denoise_strength)
print("Saved images at:")
for path in saved_paths:
    print(path)

