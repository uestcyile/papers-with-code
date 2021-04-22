import torch

from squeezenet import squeezenet1_0, squeezenet1_1

# create model
model = squeezenet1_1(pretrained=True)

# create input data
x = torch.randn((2, 3, 224, 224))

# output
out = model(x)
print('end')