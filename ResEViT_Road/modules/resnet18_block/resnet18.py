from torch import nn
from torch import Tensor

class BasicBlock(nn.Module):
    def __init__(
        self, 
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expansion: int = 1,
        downsample: nn.Module = None
    ) -> None:
        super(BasicBlock, self).__init__()
        self.expansion = expansion
        self.downsample = downsample
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3, 
            stride=stride, 
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, 
            out_channels*self.expansion, 
            kernel_size=3, 
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels*self.expansion)
    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return  out

class ResNet_input(nn.Module):
    def __init__(self,img_channels=3,out_channel=64):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=img_channels,
            out_channels=out_channel,
            kernel_size=7, 
            stride=2,
            padding=3,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    def forward(self,x):
        x=self.conv(x)
        x=self.bn(x)
        x=self.relu(x)
        x=self.max_pool(x)
        return x
        
class ResNet_block(nn.Module):
    def __init__(self,in_channel,out_channel,stride=1,depth=2):
        super().__init__()
        self.expansion = 1
        self.blocks=self._make_layer(in_channel,out_channel,stride,depth)
    def _make_layer(self,in_channel: int, out_channel: int, stride: int = 1,depth: int=2):
        downsample = None
        if stride != 1:
            downsample = nn.Sequential(
                nn.Conv2d(
                    in_channel, 
                    out_channel*self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False 
                ),
                nn.BatchNorm2d(out_channel * self.expansion),
            )
        layers = []
        layers.append(
             BasicBlock(
                in_channel, out_channel, stride, self.expansion, downsample
            )
        )
        in_channel = out_channel * self.expansion
        for i in range(1, depth):
            layers.append(BasicBlock(
                in_channel,
                out_channel,
                expansion=self.expansion
            ))
        return nn.Sequential(*layers)
    def forward(self,x):
        x=self.blocks(x)
        return x