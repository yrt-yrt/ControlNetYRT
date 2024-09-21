import numpy as np
import torch


control_num_str = '5'
control_num = [int(char) for char in control_num_str]
control = torch.tensor([control_num])
control = control.to('cuda:0')

print(control.size())
