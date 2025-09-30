import torch.nn as nn


class vgg16_conv_block(nn.Module):
    def __init__(self, input_channels, out_channels, rate=0.4, drop=True):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, out_channels, 3, 1, 1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(rate)
        self.drop = drop

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        if self.drop:
            x = self.dropout(x)
        return (x)


def vgg16_layer(input_channels, out_channels, num, dropout=[0.4, 0.4]):
    result = []
    result.append(vgg16_conv_block(input_channels, out_channels, dropout[0]))
    for i in range(1, num - 1):
        result.append(vgg16_conv_block(out_channels, out_channels, dropout[1]))
    if num > 1:
        result.append(vgg16_conv_block(out_channels, out_channels, drop=False))
    result.append(nn.MaxPool2d(2, 2))
    return (result)


class VGG16(nn.Module):
    def __init__(self):
        super().__init__()
        self.b1 = nn.Sequential(
            *vgg16_layer(3, 64, 2, [0.3, 0.4]),
            *vgg16_layer(64, 128, 2),
            *vgg16_layer(128, 256, 3),
            *vgg16_layer(256, 512, 3),
            *vgg16_layer(512, 512, 3)
        )
        self.b2 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Flatten(),
            nn.Linear(512, 512, bias=True),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 10, bias=True)
        )

    def forward(self, x):
        x = nn.Sequential(self.b1, self.b2)(x)
        return x

