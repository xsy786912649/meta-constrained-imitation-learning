import torch
x = torch.tensor([[1.0,2.0], [2.0,3.0]], requires_grad=True)
print(x)
y = x*x*x
y1 = x*x
v = torch.ones(x.shape,dtype=torch.float) 
y.backward(v,retain_graph=True, create_graph=True)
x1=x.grad
print(x1)
y1.backward(v,retain_graph=True, create_graph=True)
x2=x.grad
print(x2)

vel1=torch.autograd.grad(y,x,v,retain_graph=True, create_graph=True)
vel2=torch.autograd.grad(y1,x,v,retain_graph=True, create_graph=True)
print(vel1[0])
print(vel2[0])