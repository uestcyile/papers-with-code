import torch

from googlenet import googlenet

device = 'cpu'
model = googlenet(pretrained=False, num_classes=1000, aux_logits=True, transform_input=True,
    init_weights=True)
# model.eval()
model.to(device=device)

x = torch.randn((1, 3, 224, 224)).to(device=device)
out, aux1, aux2 = model(x)
print('end')
