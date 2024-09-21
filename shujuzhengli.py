import numpy as np
import re
# 初始化两个空列表来存储 loss 和 control 数据
loss_list = []
control_list = []
backward_list = []
# 打开文件并逐行读取
with open('jieguojiluv3minus4_minus2.txt', 'r') as file:
    for line in file:
        if 'loss :' in line:
            # 提取 loss 值
            loss_value = float(line.split('loss :')[-1].strip())
            loss_list.append(loss_value)
        elif 'control :' in line:
            # 提取 control 值
            #control_str = line.split('control :tensor([')[-1].split(']')[0].strip()
            #print(control_str)
            #control_value = float(control_str)
            #control_list.append(control_value)
            match = re.search(r'control :tensor\(\[\[([-0-9.]+)\]\], requires_grad=True\)', line)
            if match:
                control_value = float(match.group(1))
                control_list.append(control_value)
        elif 'behind backword:' in line:
            match2 = re.search(r'behind backword: tensor\(\[\[([-0-9.]+)\]\]\)', line)
            if match2:
                backward_value = float(match2.group(1))
                backward_list.append(backward_value)

# 将列表转换为 numpy 数组
loss_array = np.array(loss_list)
control_array = np.array(control_list)
backward_array = np.array(backward_list)

print("Loss array:", loss_array)
print("Control array:", control_array)
print("Backward array:", backward_array)
print("Loss array:", ", ".join(map(str, loss_array)))
print("Control array:", ", ".join(map(str, control_array)))
print("Backward array:", ", ".join(map(str, backward_array)))
