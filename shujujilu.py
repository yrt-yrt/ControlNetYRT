import pandas as pd
import re
# 读取文本文件
filename = 'jiluyixia.txt'
lines = []
with open(filename, 'r') as file:
    lines = file.readlines()

# 初始化存储数据的列表
data = {
    'behind backword': [],
    'loss': [],
    'control': []
}

number_pattern = r'[-+]?\d*\.\d+|\d+'  # 匹配浮点数或整数

# 提取数据
for line in lines:
    if line.startswith('behind backword:'):
        matches = re.findall(number_pattern, line)
        behind_backword = float(matches[-1])
        data['behind backword'].append(behind_backword)
    elif line.startswith('loss :'):
        matches = re.findall(number_pattern, line)
        loss = float(matches[-1])
        data['loss'].append(loss)
    elif line.startswith('control :'):
        matches = re.findall(number_pattern, line)
        control = float(matches[-1])
        data['control'].append(control)
# 创建DataFrame
df = pd.DataFrame(data)

# 保存为Excel文件
excel_filename = 'output_data.xlsx'
df.to_excel(excel_filename, index=False)

print(f'Data saved to {excel_filename}')
   

