from shufflenetv2 import shufflenet_v2_x1_0
import torch

model = shufflenet_v2_x1_0(pretrained=True)
x = torch.randn((1, 3, 224, 224))
out = model(x)
print('end')