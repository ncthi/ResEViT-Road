from .modules import  ResNet_input, ResNet_block, EfficientViT_input, EfficientViT_block
from torch import nn
from .layers.nn import ConcatLayer,InterleavedLayer,InverseResidualBlock
import torch
import torch.nn.functional as F




class Combine_block(nn.Module):
    def __init__(self,x_channel,y_channel):
        super().__init__()
        self.concat_layer=ConcatLayer()
        self.MB_block=InverseResidualBlock(in_channels=(x_channel+y_channel),out_channels=512,kernel_size=3)
    def forward(self,x,y):
        result=self.concat_layer(x,y)
        result=self.MB_block(result)
        return result

class Interactive_block(nn.Module):
    def __init__(self,x_channel,y_channel):
        super().__init__()
        self.in_channels=x_channel//2+y_channel//2
        self.down_channel1=nn.Conv2d(in_channels=x_channel,out_channels=x_channel//2,kernel_size=1)
        self.down_channel2=nn.Conv2d(in_channels=y_channel,out_channels=y_channel//2,kernel_size=1)
        self.mlp=nn.Sequential(
            nn.Linear(self.in_channels,self.in_channels*4),
            nn.ReLU(inplace=True),
            nn.Linear(self.in_channels*4,self.in_channels),
        )
    def forward(self,x,y):
        B, _, H,W = x.shape  # Batch size, height, width
        xt=self.down_channel1(x)
        yt=self.down_channel2(y)
        result=torch.cat((xt,yt),dim=1)
        result=result.reshape(B, self.in_channels, -1).permute(0, 2, 1)
        result=self.mlp(result).permute(0, 2, 1).reshape(B, self.in_channels, H, W)
        return result+x,result+y

class CrossAttention(nn.Module):
    def __init__(self,x_channel,y_channel,image_size,out_channel=256):
        super(CrossAttention, self).__init__()
        self.out_d=out_channel
        self.q =nn.Sequential(
            nn.Conv2d(in_channels=x_channel,out_channels=x_channel,kernel_size=3,stride=1,padding=1,groups=x_channel),
            nn.Conv2d(in_channels=x_channel,out_channels=out_channel,kernel_size=1),
        )
        self.k = nn.Sequential(
            nn.Conv2d(in_channels=y_channel,out_channels=y_channel,kernel_size=3,stride=1,padding=1,groups=y_channel),
            nn.Conv2d(in_channels=y_channel,out_channels=out_channel,kernel_size=1),
        )
        self.v = nn.Sequential(
            nn.Conv2d(in_channels=x_channel,out_channels=x_channel,kernel_size=3,stride=1,padding=1,groups=x_channel),
            nn.Conv2d(in_channels=x_channel,out_channels=out_channel,kernel_size=1),
        )
        self.relu=nn.ReLU(inplace=True)

    def forward(self,x,y):
        B, _, H, W = x.shape  # Batch size, height, width

        q = self.q(x).reshape(B, self.out_d, -1).permute(0, 2, 1)  # [B, HW, C]
        k = self.k(y).reshape(B, self.out_d, -1)  # [B, C, HW]
        v=self.v(x).reshape(B, self.out_d, -1).permute(0, 2, 1)

        attn_weights = torch.matmul(q, k) / (self.out_d ** 0.5)
        attn_weights = F.softmax(attn_weights, dim=-1)

        attn_out = torch.matmul(attn_weights, v).permute(0, 2, 1).reshape(B, self.out_d, H, W)
        attn_out = self.relu(attn_out)
        return attn_out

class ResEViT_road_backbone(nn.Module):
    def __init__(self,
                 res_channels: list[int] = [32, 32,64,128, 256],
                 res_depths: list[int] = [1, 2, 2, 2],
                 efficientViT_channels: list[int] = [32,32,64, 128, 256],
                 efficientViT_depths: list[int] = [1, 1, 2, 2]) -> None:

        super().__init__()
        self.resnet_input=ResNet_input(img_channels=3,out_channel=res_channels[0])
        self.efficientViT_input=EfficientViT_input(3,efficientViT_channels[0],2)
        self.resnet_blocks=nn.ModuleList()
        self.efficientViT_blocks=nn.ModuleList()
        self.interactive_blocks=nn.ModuleList()
        res_out_channel=res_channels[0]
        vit_out_channel=efficientViT_channels[0]
        for i,(res_channel,vit_channel,res_depth,vit_depth) in enumerate(zip(res_channels[1:],efficientViT_channels[1:],res_depths,efficientViT_depths)):
            self.resnet_blocks.append(ResNet_block(res_out_channel,res_channel,stride=2, depth=res_depth))
            self.efficientViT_blocks.append(EfficientViT_block(vit_out_channel,vit_channel,depth=vit_depth))
            self.interactive_blocks.append(Interactive_block(res_channel,vit_channel))
            res_out_channel=res_channel
            vit_out_channel=vit_channel
        self.combine_block=CrossAttention(res_out_channel,vit_out_channel,4)

    def forward(self,x):
        res=self.resnet_input(x)
        vit=self.efficientViT_input(x)
        for res_block,efficientViT_block,interactive_block in zip(self.resnet_blocks,self.efficientViT_blocks,self.interactive_blocks):
            res=res_block(res)
            vit=efficientViT_block(vit)
            res,vit=interactive_block(res,vit)
        x=self.combine_block(res,vit)
        return x

class ResEViT_road_cls(nn.Module):
    def __init__(self,num_classes=2):
        super().__init__()
        self.baseModel=ResEViT_road_backbone()
        self.adaptive=nn.AdaptiveAvgPool2d(1)
        self.classification=nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.1),
            nn.ReLU(inplace=True),
            nn.Linear(256,num_classes)
        )
    def forward(self,x):
        x=self.baseModel(x)
        x=self.adaptive(x)
        x=self.classification(x)
        return x




