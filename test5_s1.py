from share import *
import config

import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random
from torch.optim import LBFGS
#from torchviz import make_dot

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
def process(z_enc, t_enc, control, prompt, a_prompt, n_prompt, num_samples, guess_mode, strength, scale, seed):
    global preprocessor

    #with torch.no_grad():
    #input_image = HWC3(input_image)
        
    #img = cv2.resize(input_image, (320,320))
        
    #H, W, C = img.shape

    #img = torch.from_numpy(img.copy()).float().cuda() / 127.0 - 1.0
    #img = img.requires_grad_()
    #img = torch.stack([img for _ in range(num_samples)], dim=0)
    #img = einops.rearrange(img, 'b h w c -> b c h w').clone()

    if seed == -1:
        seed = random.randint(0, 65535)
    seed_everything(seed)

    if config.save_memory:
        model.low_vram_shift(is_diffusing=False)

    cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
    un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}

    if config.save_memory:
        model.low_vram_shift(is_diffusing=False)

    #ddim_sampler.make_schedule(ddim_steps, ddim_eta=eta, verbose=True)
    #t_enc = min(int(denoise_strength * ddim_steps), ddim_steps - 1)
    #z = model.get_first_stage_encoding(model.encode_first_stage(img))
    #print(f'z:{z.shape}')
    #z_enc = ddim_sampler.stochastic_encode(z, torch.tensor([t_enc] * num_samples).to(model.device))
    #print(f'z_enc:{z_enc.shape}')
    if config.save_memory:
        model.low_vram_shift(is_diffusing=True)

    model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)
        # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01

    samples = ddim_sampler.decode(z_enc, cond, t_enc, unconditional_guidance_scale=scale, unconditional_conditioning=un_cond)
    print(f'samples:{samples.shape}')
    #if config.save_memory:
        #model.low_vram_shift(is_diffusing=False)

    #x_samples = model.decode_first_stage(samples)
    #x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

    #results = [x_samples[i] for i in range(num_samples)]
    return samples

input_image_path = '/workspace/home/yuanrt/ControlNet1_1/ControlNet-v1-1-nightly/00013-0-bag.png'
input_image = cv2.imread(input_image_path)
target_image_path = '/workspace/home/yuanrt/ControlNet1_1/ControlNet-v1-1-nightly/dataForTrain/resultData/cropped_minus2_bag.png'
target_image = cv2.imread(target_image_path)

control = torch.tensor([[-4]], requires_grad = True, dtype=torch.float32)#.cuda()
#target = torch.tensor(target_image, dtype=torch.float32, requires_grad = True).cuda()
targets = torch.from_numpy(target_image.copy()).float().cuda() / 127.0 - 1.0
targets = targets.requires_grad_()
targets = torch.stack([targets for _ in range(1)], dim=0)
targets = einops.rearrange(targets, 'b h w c -> b c h w').clone()
target = model.get_first_stage_encoding(model.encode_first_stage(targets)).cuda()
target.requires_grad_(True)

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

input_image = HWC3(input_image)

img = cv2.resize(input_image, (320,320))

H, W, C = img.shape

img = torch.from_numpy(img.copy()).float().cuda() / 127.0 - 1.0
img = img.requires_grad_()
img = torch.stack([img for _ in range(num_samples)], dim=0)
img = einops.rearrange(img, 'b h w c -> b c h w').clone()

z = model.get_first_stage_encoding(model.encode_first_stage(img))
#for param in model.parameters():
    #jjparam.requires_grad = False

loss_fn = torch.nn.MSELoss()

control = control.requires_grad_()
optimizer = torch.optim.SGD([control], lr=0.35)
#param_name = "control_model.middle_block.1.transformer_blocks.0.norm1.weight"
#state_dict = model.state_dict()
#found_param = None
#for name, param in model.named_parameters():
    #if name == param_name:
        #found_param = param
        #break
rs = torch.zeros((1, 4, 40, 40))
for i in range(300):
    print(f'{i} turn')
    optimizer.zero_grad()
    #control.grad.zero_()
    #control = control.requires_grad_()
    control_s = control.cuda()
    if(i == 0):
        ddim_sampler.make_schedule(ddim_steps, ddim_eta=eta, verbose=True)
        t_enc = min(int(denoise_strength * ddim_steps), ddim_steps - 1)
        z_enc = ddim_sampler.stochastic_encode(z, torch.tensor([t_enc] * num_samples).to(model.device))
    else:
        ddim_sampler.make_schedule(9, ddim_eta=eta, verbose=True)
        t_enc = min(int(denoise_strength * 9), 9 - 1)
        z_enc = ddim_sampler.stochastic_encode(rs, torch.tensor([t_enc] * num_samples).to(model.device))
    result = process(z_enc, t_enc, control_s, prompt, a_prompt, n_prompt, num_samples, guess_mode, strength, scale, seed)
    #res = result[0]
    #pred = torch.tensor(res, dtype=torch.float32, requires_grad = True).cuda()
    #print(target.shape)
    #print(pred.shape)
    rs = result
    loss = loss_fn(target, result)
    print(f'front backword: {control.grad}')
    #print(found_param.grad)
    #print(state_dict[param_name])
    #control.retain_grad()
    
    
    
    loss.backward(retain_graph=True)
    #print(middle_block.1.transformer_blocks.0.attn1.to_out.0.bias.grad)
    print(f'behind backword: {control.grad}')
    #print(model.state_dict()[param_name + "_grad"])
    #dot = make_dot(loss, params={'control':control})
    
    #dot.render('control_gradient', format='png')
    #print(found_param.grad)
    #print(state_dict[param_name])
    #print(control)
    optimizer.step()
    #control = control - control.grad*0.79
    print(f'loss :{loss}')
    #control = control.detach().clone().requires_grad_()
    print(f'control :{control}')
    #control.grad.zero_()
    #print(found_param.grad)
    #print(state_dict[param_name])
    #print(model.state_dict()[param_name + "_grad"])
    #print(middle_block.1.transformer_blocks.0.attn1.to_out.0.bias.grad)
    #print(loss)
    #print(control)
   

