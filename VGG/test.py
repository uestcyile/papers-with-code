import torch
from vgg import vgg16, vgg16_bn

device = 'cpu'
vgg_model = vgg16_bn(pretraind=True, progress=True, num_classes=1000)
vgg_model.eval().to(device=device)

x = torch.rand((1, 3, 224, 224)).to(device=device)
out = vgg_model(x)
print('end')



