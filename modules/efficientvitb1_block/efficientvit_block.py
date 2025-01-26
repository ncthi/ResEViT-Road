import torch
import torch.nn as nn

from .nn import (
    ConvLayer,
    DSConv,
    EfficientViTBlock,
    FusedMBConv,
    IdentityLayer,
    MBConv,
    OpSequential,
    ResBlock,
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
    
class input_stem(nn.Module):
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
    def forward(self,x):
        x = self.input_stem(x)
        return x

class MBConv_stage(nn.Module):
    def __init__(self,in_channel,out_channel,depth,expand_ratio=4,norm="bn2d",act_func="hswish"):
        super().__init__()
        self.stage=[]
        for i in range(depth):
            stride = 2 if i == 0 else 1
            block = build_local_block(
                in_channels=in_channel,
                out_channels=out_channel,
                stride=stride,
                expand_ratio=expand_ratio,
                norm=norm,
                act_func=act_func,
            )
            block = ResidualBlock(block, IdentityLayer() if stride == 1 else None)
            self.stage.append(block)
            in_channel = out_channel
        self.stage=OpSequential(self.stage)
    def forward(self,x):
        x = self.stage(x)
        return x
        
class EfficientViT_stage(nn.Module):
    def __init__(self,in_channel,out_channel,depth,expand_ratio=4,norm="bn2d",act_func="hswish", dim=16):
        super().__init__()
        self.stage=[]
        block = build_local_block(
            in_channels=in_channel,
            out_channels=out_channel,
            stride=2,
            expand_ratio=expand_ratio,
            norm=norm,
            act_func=act_func,
            fewer_norm=True,
        )
        self.stage.append(ResidualBlock(block, None))
        in_channel = out_channel
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
    def forward(self,x):
        x = self.stage(x)
        return x
        
        