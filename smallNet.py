import torch

x_in=torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]]).view((1,1,3,3)).type(torch.FloatTensor)
y_out=torch.tensor([[19, 25],[37, 43]]).view(1,1,2,2).type(torch.FloatTensor)
c_core=torch.randn(2,2).view((1,1,2,2)).type(torch.FloatTensor)
c_core=c_core.requires_grad_()    
LR=0.01
# 定义一个损失函数
loss_fun=torch.nn.MSELoss()
for i in range(10):
    y_pre=torch.nn.functional.conv2d(x_in,c_core)
    loss=loss_fun(y_pre,y_out)
    print(c_core.grad)
    c_core.retain_grad()
    loss.backward()
    c_core=c_core-c_core.grad*LR
    print('the loss is:',loss)
print('c_core: ',c_core)
