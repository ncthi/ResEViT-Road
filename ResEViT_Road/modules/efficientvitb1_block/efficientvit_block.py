import torch.nn as nn

from .nn import (
    ConvLayer,
    DSConv,
    EfficientViTBlock,
    IdentityLayer,
    MBConv,
    OpSequential,
    ResidualBlock,
)


def build_local_block(
    in_channels: int,
    out_channels: int,
    stride: int,
    expand_ratio: float,
    norm: str,
    act_func: str,
    fewer_norm: bool = False,
    ) -> nn.Module:
    if expand_ratio == 1:
        block = DSConv(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            use_bias=(True, False) if fewer_norm else False,
            norm=(None, norm) if fewer_norm else norm,
            act_func=(act_func, None),
        )
    else:
        block = MBConv(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            expand_ratio=expand_ratio,
            use_bias=(True, True, False) if fewer_norm else False,
            norm=(None, None, norm) if fewer_norm else norm,
            act_func=(act_func, act_func, None),
        )
    return block
    
class EfficientViT_input(nn.Module):
    def __init__(self,in_channel,out_channel,depth,expand_ratio=4,norm="bn2d",act_func="hswish"):
        super().__init__()
        self.input_stem = [
            ConvLayer(
                in_channels=in_channel,
                out_channels=out_channel,
                stride=2,
                norm=norm,
                act_func=act_func,
            )
        ]
        for _ in range(depth):
            block = build_local_block(
                in_channels=out_channel,
                out_channels=out_channel,
                stride=1,
                expand_ratio=1,
                norm=norm,
                act_func=act_func,
            )
            self.input_stem.append(ResidualBlock(block, IdentityLayer()))
        self.input_stem = OpSequential(self.input_stem)
        self.down_sample=nn.Conv2d(out_channel,out_channel,3,2,1)
    def forward(self,x):
        x = self.input_stem(x)
        x = self.down_sample(x)
        return x
class EfficientViT_block(nn.Module):
    def __init__(self,in_channel,out_channel,depth,expand_ratio=4,norm="bn2d",act_func="hswish", dim=16):
        super().__init__()
        self.stage=[]
        for _ in range(depth):
            self.stage.append(
                EfficientViTBlock(
                    in_channels=in_channel,
                    dim=dim,
                    expand_ratio=expand_ratio,
                    norm=norm,
                    act_func=act_func,
                )
            )
        self.stage=OpSequential(self.stage)
        self.pw_conv=nn.Conv2d(in_channel,out_channel,kernel_size=1)
        self.down_sample=nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    def forward(self,x):
        x = self.stage(x)
        x = self.pw_conv(x)
        x = self.down_sample(x)
        return x
        
        