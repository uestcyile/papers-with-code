import torch
from resnet import resnet50, resnext50_32x4d

device = 'cpu'

#=================resnet50=================
model_resnet50 = resnet50(pretrained=True)
model_resnet50.eval().to(device=device)

# input
x = torch.randn((1, 3, 224, 224)).to(device=device)
out1 = model_resnet50(x)


#================resnext50_32x4d================
model_resnext50 = resnext50_32x4d(pretrained=True)
model_resnext50.eval().to(device=device)

out2 = model_resnext50(x)
print('end')