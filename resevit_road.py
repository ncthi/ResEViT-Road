from .modules import  input_stem, MBConv_stage, EfficientViT_stage,resnet_block,resnet_in
from torch import nn
from .layers.nn import ConcatLayer,InterleavedLayer, CascadedGroupAttention, LocalWindowAttention
from .modules.efficientvitb1_block.nn import LiteMLA
import torch.nn.functional as F

class Combine_block(nn.Module):
    def __init__(self,x_channel,y_channel):
        super().__init__()
        self.concat_layer=ConcatLayer()
        out_channel=x_channel+y_channel
        self.dw_conv=nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=1,groups=out_channel)
        self.point_conv=nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=1)
        self.nor=nn.LayerNorm([out_channel,7,7])
    def forward(self,x,y):
        result=self.concat_layer(x,y)
        result=self.dw_conv(result)
        result=self.point_conv(result)
        result=self.nor(result)
        return result
        
class Interactive_block(nn.Module):
    def __init__(self,x_channel,y_channel,out_channel,index):
        super().__init__()
        self.interleavedLayer=InterleavedLayer(x_channel,y_channel)
        self.attn= CascadedGroupAttention(384, 16, num_heads=4,attn_ratio=6,resolution=14,kernels=[5, 5, 5, 5])
        self.downchannel=nn.Conv2d(x_channel+y_channel,out_channel,kernel_size=1)
        self.relu=nn.ReLU()
        self.bn = nn.BatchNorm2d(out_channel)
        self.index=index
    def forward(self,x,y):
        result= self.interleavedLayer(x,y)
        result=self.attn(result)
        result=self.downchannel(result)
        if self.index==0: result=x+result
        else:  result=y+result
        result=self.relu(result)
        result= self.bn(result)
        return result
        

class restevit_road_backbone(nn.Module):
    def __init__(self,
        res_channels=[64,128,256,512],
        res_depths=[2,2,2,2],
        efficientvit_channels=[16, 32, 64, 128, 256],
        efficientvit_depths=[1, 2, 3, 3, 4]):
        super().__init__()
        self.resnet_in=resnet_in()
        self.efficientvit_in=input_stem(3,efficientvit_channels[0],efficientvit_depths[0])
        self.block1=resnet_block(64,res_channels[0],1,res_depths[0])  
        self.block2=resnet_block(res_channels[0],res_channels[1],2,res_depths[1])      
        self.block3=resnet_block(res_channels[1],res_channels[2],2,res_depths[2])      
        self.block4=resnet_block(res_channels[2],res_channels[3],2,res_depths[3])  
        self.stage1=MBConv_stage(efficientvit_channels[0],efficientvit_channels[1],efficientvit_depths[1])
        self.stage2=MBConv_stage(efficientvit_channels[1],efficientvit_channels[2],efficientvit_depths[2])
        self.stage3=EfficientViT_stage(efficientvit_channels[2],efficientvit_channels[3],efficientvit_depths[3])
        self.stage4=EfficientViT_stage(efficientvit_channels[3],efficientvit_channels[4],efficientvit_depths[4])
        self.combine_block=Combine_block(res_channels[3],efficientvit_channels[4])
        self.interactive_block_to_r=Interactive_block(res_channels[2],efficientvit_channels[3],res_channels[2],0)
        self.interactive_block_to_v=Interactive_block(res_channels[2],efficientvit_channels[3],efficientvit_channels[3],1)

    def forward(self,x):
        res=self.resnet_in(x)
        vit=self.efficientvit_in(x)
        res=self.block1(res)
        vit=self.stage1(vit)
        res=self.block2(res)
        vit=self.stage2(vit)
        res=self.block3(res)
        vit=self.stage3(vit)
        res=self.interactive_block_to_r(res,vit)
        vit=self.interactive_block_to_v(res,vit)
        res=self.block4(res)
        vit=self.stage4(vit)
        result=self.combine_block(res,vit)
        return result

class restevit_road_cls(nn.Module):
    def __init__(self,num_class=2):
        super().__init__()
        self.base_mdoel=restevit_road_backbone()
        self.apdative=nn.AdaptiveAvgPool2d(1)
        self.classification=nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(768,num_class)
        )
    def forward(self,x):
        x=self.base_mdoel(x)
        x=self.apdative(x)
        x=self.classification(x)
        return x




