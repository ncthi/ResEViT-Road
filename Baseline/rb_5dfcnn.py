import torch
import torch.nn as nn
import torch.nn.functional as F

# ======== Block phụ trợ: Convolutional Block & Identity Block (Fig. 3) ========
class ConvBlock(nn.Module):
    def __init__(self, in_channels, filters):
        super(ConvBlock, self).__init__()
        f1, f2, f3 = filters
        self.conv1 = nn.Conv2d(in_channels, f1, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(f1)
        self.conv2 = nn.Conv2d(f1, f2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(f2)
        self.conv3 = nn.Conv2d(f2, f3, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(f3)
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, f3, kernel_size=1),
            nn.BatchNorm2d(f3)
        )

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x += shortcut
        return F.relu(x)


class IdentityBlock(nn.Module):
    def __init__(self, in_channels, filters):
        super(IdentityBlock, self).__init__()
        f1, f2, f3 = filters
        self.conv1 = nn.Conv2d(in_channels, f1, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(f1)
        self.conv2 = nn.Conv2d(f1, f2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(f2)
        self.conv3 = nn.Conv2d(f2, f3, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(f3)

    def forward(self, x):
        shortcut = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x += shortcut
        return F.relu(x)

# ======== Khối tổng thể RB-5DFCNN ========
class RB5DFCNN(nn.Module):
    def __init__(self,num_classes=2):
        super(RB5DFCNN, self).__init__()

        self.zero_pad = nn.ZeroPad2d(3)  # như ResNet50

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Block 2
        self.block2 = self._make_block(64, [64, 64, 256], num_identity=2)

        # Block 3
        self.block3 = self._make_block(256, [128, 128, 512], num_identity=3)

        # Block 4
        self.block4 = self._make_block(512, [256, 256, 1024], num_identity=5)

        # Block 5
        self.block5 = self._make_block(1024, [512, 512, 2048], num_identity=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

        # Deep Fully Connected Network (Fig. 4)
        self.fc_layers = nn.Sequential(
            nn.Linear(2048, 500),
            nn.ReLU(),
            nn.Linear(500, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, num_classes),  # Assuming 13 classes
        )

    def _make_block(self, in_channels, filters, num_identity):
        layers = [ConvBlock(in_channels, filters)]
        for _ in range(num_identity):
            layers.append(IdentityBlock(filters[2], filters))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.zero_pad(x)
        x = self.conv1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc_layers(x)
        return x
