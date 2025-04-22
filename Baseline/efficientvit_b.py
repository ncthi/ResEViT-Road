import timm
from torch import nn

class EfficientViT_B1(nn.Module):
    def __init__(self,num_classes=2):
        super(EfficientViT_B1, self).__init__()
        self.model = timm.create_model('efficientvit_b1', pretrained=False, num_classes=num_classes)
    def forward(self, x):
        return self.model(x)

class EfficientViT_B2(nn.Module):
    def __init__(self,num_classes=2):
        super(EfficientViT_B2, self).__init__()
        self.model = timm.create_model('efficientvit_b2', pretrained=False, num_classes=num_classes)
    def forward(self, x):
        return self.model(x)