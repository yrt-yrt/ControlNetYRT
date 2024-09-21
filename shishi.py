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
model_state_dict = model.state_dict()

output_file_path = 'model_output.txt'

with open(output_file_path, 'w') as file:
    # 重定向 print 输出到文件
    print(f"Model State Dict:\n", file=file)

    for key, value in model_state_dict.items():
        print(f"Key: {key}, Shape: {value.shape}", file=file)
#�~J| ��~F��~@个control_num_str


   

