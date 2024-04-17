import torch
import warnings
import torch.nn as nn


torch.manual_seed(42)
warnings.simplefilter("ignore")


class ResNet9(nn.Module):
    def __init__(self):
        super().__init__()
        # in 32*32*3
        self.conv1 = self.conv_block(3, 64)
        self.conv2 = self.conv_block(64, 128, True)

        self.res1 = nn.Sequential(self.conv_block(128, 128), self.conv_block(128, 128))
        self.conv3 = self.conv_block(128, 256, True)
        self.conv4 = self.conv_block(256, 512, True)
        self.res2 = nn.Sequential(self.conv_block(512, 512), self.conv_block(512, 512))

        self.classifier = nn.Sequential(
            nn.MaxPool2d(4),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(512, 10),
        )

    def conv_block(self, in_ch, out_ch, pool=False):
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2, 2))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out
