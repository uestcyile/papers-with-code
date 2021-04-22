from torch import nn

from utils import load_state_dict_from_url

__all__ = ['MobileNetV2', 'mobilenet_v2']

model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1,
                 groups=1, norm_layer=None):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        padding = kernel_size // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            norm_layer(out_planes),
            nn.ReLU6(inplace=True)
        )

class InvertedResidual(nn.Module):
    def __init__(self, in_planes, out_planes, stride, expand_ratio, norm_layer=None):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        hidden_dim = int(round(in_planes * expand_ratio))
        self.use_res_connect = (self.stride == 1 and in_planes == out_planes)

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(in_planes, hidden_dim, kernel_size=1, norm_layer=norm_layer))
        layers.extend([
            # depthwise
            ConvBNReLU(hidden_dim, hidden_dim, kernel_size=3, stride=stride, groups=hidden_dim, norm_layer=norm_layer),
            # pointwise-linear
            nn.Conv2d(hidden_dim, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
            norm_layer(out_planes),
        ])
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, inverted_residual_setting=None,
        block=None, norm_layer=None):
        super(MobileNetV2, self).__init__()

        if block is None:
            block = InvertedResidual
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]
        
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))
        
        input_channel = 32
        last_channel = 1280

        layers = []
        # building first layer
        layers.append(ConvBNReLU(3, input_channel, kernel_size=3, stride=2, norm_layer=norm_layer))
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            for i in range(n):
                stride = s if i == 0 else 1
                layers.append(block(input_channel, c, stride, t, norm_layer))
                input_channel = c
        # building last several layers
        layers.append(ConvBNReLU(input_channel, last_channel, kernel_size=1, norm_layer=norm_layer))
        # make it nn.Sequential
        self.features = nn.Sequential(*layers)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, num_classes)
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
        
    def forward(self, x):
        out = self.features(x)
        out = nn.functional.adaptive_avg_pool2d(out, output_size=1).reshape(x.shape[0], -1)
        out = self.classifier(out)
        return out

def mobilenet_v2(pretrained=False, progress=True, **kwargs):
    model = MobileNetV2(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['mobilenet_v2'], progress=progress)
        model.load_state_dict(state_dict)
    
    return model