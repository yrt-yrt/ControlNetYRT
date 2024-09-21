import torch
"""
a = [1,4]
b = [1,5]
c1 = torch.tensor(a)
c2 = torch.tensor(b)
print(c1)
print(c2)

c = torch.cat((c1,c2),-1)
print(c)
print(c.shape)
"""
batch_size = 32
channels = 3
height = 128
width = 128
a = torch.randn(batch_size, channels, height, width)
print(a.shape)

b = torch.randn(32,3,128,128)

c = torch.cat((a,b),1)
print(c.shape)