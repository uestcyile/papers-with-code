import torch
import torch.nn as nn

from utils import load_state_dict_from_url


__all__ = [
    'ShuffleNetV2', 'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0',
    'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0'
]

model_urls = {
    'shufflenetv2_x0.5': 'https://download.pytorch.org/models/shufflenetv2_x0.5-f707e7126e.pth',
    'shufflenetv2_x1.0': 'https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth',
    'shufflenetv2_x1.5': None,
    'shufflenetv2_x2.0': None,
}

def channel_shuffle(x, groups):
    b, c, h, w = x.data.size()

    channels_per_group = c // groups

    # reshape
    x = x.view(b, groups, channels_per_group, h, w)
    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(b, -1, h, w)

    return x

# building block for shufflenet v2
class InvertedResidual(nn.Module):
    def __init__(self, in_planes, out_planes, stride):
        # out_planes==in_planes or out_planes==2*in_planes
        super(InvertedResidual, self).__init__()

        if not (1 <= stride <= 3):
            raise ValueError('illegal stride value')
        self.stride = stride

        branch_planes = out_planes // 2

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=self.stride,
                          padding=1, bias=False, groups=in_planes),
                nn.BatchNorm2d(in_planes),
                nn.Conv2d(in_planes, branch_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_planes),
                nn.ReLU(inplace=True)
            )
        else:
            self.branch1 = nn.Sequential()
        
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_planes if (self.stride > 1) else branch_planes, branch_planes,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes, branch_planes, kernel_size=3, stride=self.stride,
                      padding=1, groups=branch_planes, bias=False),
            nn.BatchNorm2d(branch_planes),
            nn.Conv2d(branch_planes, branch_planes, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_planes),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out1 = x1
            out2 = self.branch2(x2)
            out = torch.cat((out1, out2), dim=1)
        else:
            out1 = self.branch1(x)
            out2 = self.branch2(x)
            out = torch.cat((out1, out2), dim=1)
        
        out = channel_shuffle(out, groups=2)
        return out


class ShuffleNetV2(nn.Module):
    def __init__(self, stages_repeats, stages_out_channels, num_classes=1000,
                 inveted_residual=InvertedResidual):
        super(ShuffleNetV2, self).__init__()

        if len(stages_repeats) != 3:
            raise ValueError('expected stages_repeats as list of 3 positive ints')
        if len(stages_out_channels) != 5:
            raise ValueError('expected stages_out_channels as list of 5 positive ints')
        
        in_channel = 3
        out_channel = stages_out_channels[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        in_channel = out_channel
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        stage_names = ['stage{}'.format(i) for i in [2, 3, 4]]
        for name, repeats, out_channel in zip(stage_names, stages_repeats, stages_out_channels[1:-1]):
            seq = []
            for i in range(repeats):
                stride = 2 if i == 0 else 1
                seq.append(inveted_residual(in_channel, out_channel, stride))
            setattr(self, name, nn.Sequential(*seq))
            in_channel = out_channel
        
        out_channel = stages_out_channels[-1]
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

        self.fc = nn.Linear(out_channel, num_classes)


    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        x = nn.functional.adaptive_avg_pool2d(x, output_size=1).reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

def _shufflenetv2(arch, pretrained, progress, *args, **kwargs):
    model = ShuffleNetV2(*args, **kwargs)

    if pretrained:
        model_url = model_urls[arch]
        state_dict = load_state_dict_from_url(model_url, progress=progress)
        model.load_state_dict(state_dict)

    return model

def shufflenet_v2_x0_5(pretrained=False, progress=True, **kwargs):
    return _shufflenetv2('shufflenetv2_x0.5', pretrained, progress,
                         [4, 84, 4], [24, 48, 96, 192, 1024], **kwargs)


def shufflenet_v2_x1_0(pretrained=False, progress=True, **kwargs):
    return _shufflenetv2('shufflenetv2_x1.0', pretrained, progress,
                         [4, 8, 4], [24, 116, 232, 464, 1024], **kwargs)


def shufflenet_v2_x1_5(pretrained=False, progress=True, **kwargs):
    return _shufflenetv2('shufflenetv2_x1.5', pretrained, progress,
                         [4, 8, 4], [24, 176, 352, 704, 1024], **kwargs)


def shufflenet_v2_x2_0(pretrained=False, progress=True, **kwargs):
    return _shufflenetv2('shufflenetv2_x2.0', pretrained, progress,
                         [4, 8, 4], [24, 244, 488, 976, 2048], **kwargs)
