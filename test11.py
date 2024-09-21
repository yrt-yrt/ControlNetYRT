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
from tqdm import tqdm

from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps
from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler


model_name = 'control_v11f1e_sd15_tile'
model = create_model(f'./models/{model_name}.yaml').cpu()
#model.load_state_dict(load_state_dict('./models/v1-5-pruned.ckpt', location='cuda'), strict=False)
#torch.cuda.empty_cache()
model.load_state_dict(load_state_dict(f'./models/epoch=276-step=27700.ckpt'), strict=False) #�~\~I�~T��~J�1
model = model.cuda()
ddim_sampler = DDIMSampler(model)

#�~J| ��~F��~@个control_num_str
def process(input_image, control, prompt, a_prompt, n_prompt, num_samples, ddim_steps, guess_mode, strength, scale, seed, eta, denoise_strength):
    global preprocessor

    with torch.no_grad():
        input_image = HWC3(input_image)

        img = cv2.resize(input_image, (320,320))

        H, W, C = img.shape

        img = torch.from_numpy(img.copy()).float().cuda() / 127.0 - 1.0
        #img = img.requires_grad_()
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
        z_enc = z_enc.requires_grad_()
        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)
		# Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
	    #ts = torch.full((z_enc.shape[0],), 2, device=z_enc.device, dtype=torch.long)
	    #results = ddim_sampler.p_sample_ddim(z_enc, ts, cond)
	    #samples, _  = p_sample_ddim(model, z_enc, cond, ts, 5) 
        samples = decode(model, z_enc, cond, t_enc, unconditional_guidance_scale=scale, unconditional_conditioning=un_cond)
	    #samples = model.apply_model(z_enc, ts, cond)
	    
        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
        saved_paths = []
        for i, result in enumerate(results):
            save_path = f'result_1_2.jpg'  # 可以根据需要修改文件名和格式
            cv2.imwrite(save_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))  # 注意颜色通道顺序可能需要转换
            saved_paths.append(save_path)
    return saved_paths

def p_sample_ddim(model, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                  temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                  unconditional_guidance_scale=1., unconditional_conditioning=None,
                  dynamic_threshold=None):
    b, *_, device = *x.shape, x.device

    if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
        model_output = model.apply_model(x, t, c)
    else:
        model_t = model.apply_model(x, t, c)
        model_uncond = model.apply_model(x, t, unconditional_conditioning)
        model_output = model_uncond + unconditional_guidance_scale * (model_t - model_uncond)

    e_t = model_output
    
    """
    alphas = model.alphas_cumprod #if use_original_steps else self.ddim_alphas
    alphas_prev = model.alphas_cumprod_prev #if use_original_steps else self.ddim_alphas_prev
    sqrt_one_minus_alphas = model.sqrt_one_minus_alphas_cumprod #if use_original_steps else self.ddim_sqrt_one_minus_alphas
    sigmas = 0 * torch.sqrt(
            (1 - model.alphas_cumprod_prev) / (1 - model.alphas_cumprod) * (
                        1 - model.alphas_cumprod / model.alphas_cumprod_prev))#model.ddim_sigmas_for_original_num_steps #if use_original_steps else self.ddim_sigmas 
    """  
    """ 
    alphas = [0.998296, 0.96984816, 0.93758786, 0.9016043, 0.8620799, 0.81929123, 0.7736063, 0.7254777, 0.6754324,
          0.62405777, 0.571984, 0.51986456, 0.46835446, 0.41808826, 0.36965853, 0.32359615, 0.2803536, 0.24029191,
          0.20367248, 0.17065361, 0.14129217, 0.11554988, 0.09330372, 0.0743593, 0.05846652, 0.04533602, 0.03465549,
          0.02610484, 0.0193695, 0.01415114, 0.01017578, 0.00719903, 0.00500884]

    alphas_prev = [0.99914998, 0.99829602, 0.96984816, 0.93758786, 0.90160429, 0.86207992, 0.81929123, 0.7736063,
               0.7254777, 0.67543238, 0.62405777, 0.57198399, 0.51986456, 0.46835446, 0.41808826, 0.36965853,
               0.32359615, 0.28035361, 0.24029191, 0.20367248, 0.17065361, 0.14129217, 0.11554988, 0.09330372,
               0.0743593, 0.05846652, 0.04533602, 0.03465549, 0.02610484, 0.0193695, 0.01415114, 0.01017578,
               0.00719903]

    sqrt_one_minus_alphas = [0.04127926, 0.17364286, 0.24982423, 0.3136809, 0.37137592, 0.42509854, 0.47580847,
                         0.5239488, 0.56970835, 0.6131413, 0.65422934, 0.69291806, 0.7291403, 0.7628314, 0.7939405,
                         0.82243776, 0.84831977, 0.87161237, 0.89237183, 0.9106846, 0.9266649, 0.9404521, 0.952206,
                         0.96210223, 0.9703265, 0.9770691, 0.98251945, 0.9868612, 0.9902679, 0.99289924, 0.9948991,
                         0.996394, 0.99749243]

    sigmas = [0.02064835, 0.04013009, 0.12676657, 0.15602441, 0.17684747, 0.19463227, 0.21097199, 0.22650876,
          0.24154936, 0.25625679, 0.27072458, 0.28500732, 0.29913819, 0.31313618, 0.32701122, 0.34076715,
          0.35440301, 0.36791526, 0.38129727, 0.3945416, 0.40763945, 0.42058186, 0.43335977, 0.4459651,
          0.45839048, 0.47063002, 0.48267955, 0.4945367, 0.50620064, 0.51767261, 0.52895497, 0.54005149,
          0.55096662]
    """
    alphas = ddim_sampler.ddim_alphas
    alphas_prev = ddim_sampler.ddim_alphas_prev
    sqrt_one_minus_alphas = ddim_sampler.ddim_sqrt_one_minus_alphas
    sigmas = ddim_sampler.ddim_sigmas
    a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
    a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
    sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
    sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index], device=device)

    # ��~D��~K x_0
    pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()

    # 计��~W x_prev
    dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
    noise = sigma_t * torch.randn_like(x) * temperature
    x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise

    return x_prev, pred_x0

def decode(model, x_latent, cond, t_start, unconditional_guidance_scale=1.0, unconditional_conditioning=None,
               use_original_steps=False, callback=None):

    #timesteps = np.arange(ddim_sampler.ddim_timesteps) #if use_original_steps else self.ddim_timesteps
    timesteps = np.arange(ddim_sampler.ddpm_num_timesteps) if use_original_steps else ddim_sampler.ddim_timesteps
    timesteps = timesteps[:t_start]

    time_range = np.flip(timesteps)
    total_steps = timesteps.shape[0]
    print(f"Running DDIM Sampling with {total_steps} timesteps")

    iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
    x_dec = x_latent
    for i, step in enumerate(iterator):
        index = total_steps - i - 1
        ts = torch.full((x_latent.shape[0],), step, device=x_latent.device, dtype=torch.long)
        x_dec, _ = p_sample_ddim(model, x_dec, cond, ts, index=index, use_original_steps=use_original_steps, unconditional_guidance_scale=unconditional_guidance_scale, unconditional_conditioning=unconditional_conditioning)
        if callback: callback(i)
    return x_dec

input_image_path = '/home/yuanrt/ControlNet1_1/ControlNet-v1-1-nightly/00013-0-bag.png'
input_image = cv2.imread(input_image_path)
target_image_path = '/home/yuanrt/ControlNet1_1/ControlNet-v1-1-nightly/dataForTrain/resultData/cropped_4_bag.png'
target_image = cv2.imread(target_image_path)

control = torch.tensor([[1]], requires_grad = True, dtype=torch.float32).cuda()
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
#ddpm_num_timesteps = int(1000)
#alphas_cumprod = model.alphas_cumprod
#ddim_timesteps = make_ddim_timesteps(ddim_discr_method="uniform", num_ddim_timesteps=ddim_steps,
                                                  #num_ddpm_timesteps=ddpm_num_timesteps,verbose=True)
#ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   #ddim_timesteps=ddim_timesteps,
                                                                                   #eta=0,verbose=True)
#ddim_sqrt_one_minus_alphas = np.sqrt(1. - ddim_alphas)

saved_paths = process(input_image, control, prompt, a_prompt, n_prompt, num_samples, ddim_steps, guess_mode, strength, scale, seed, eta, denoise_strength)
print("Saved images at:")
for path in saved_paths:
    print(path)
"""
noise_tensor = torch.randn([1, 4, 40, 40]).cuda()
noise_tensor.requires_grad_(True)
loss_fn = torch.nn.MSELoss()

#control = control.requires_grad_()
optimizer = torch.optim.Adam([control], lr=0.6)

for i in range(5):
    #optimizer.zero_grad()
    control = control.requires_grad_()
    control_s = control.cuda()
    result = process(input_image, control_s, prompt, a_prompt, n_prompt, num_samples, ddim_steps, guess_mode, strength, scale, seed, eta, denoise_strength)
    result.requires_grad_(True)
    #print(result.shape)
    #res = result[0]
    #pred = torch.tensor(res, dtype=torch.float32, requires_grad = True).cuda()
    #print(target.shape)
    #print(pred.shape)
    loss = loss_fn(noise_tensor, result)
    #loss = result.sum()
    #optimizer.zero_grad()
    print(control.grad)
    control.retain_grad()



    loss.backward(retain_graph=True)
    print(control.grad)
    #dot = make_dot(loss, params={'control':control})
    #dot.render('control_gradients2', format='png')
    optimizer.step()
    print(loss)
    print(control)
"""

   

