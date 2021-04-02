import torch
from densenet import densenet121, densenet161, densenet169, densenet201

device = 'cpu'
model = densenet121()
model.eval().to(device=device)

x = torch.randn((1, 3, 224, 224)).to(device=device)

y = model(x)
print('end')
