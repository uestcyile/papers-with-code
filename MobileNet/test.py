import torch

from mobilenet import mobilenet_v2

# create model
model = mobilenet_v2(pretrained=True)

# create input
x = torch.randn((1, 3, 224, 224))

# output
out = model(x)
print('end')