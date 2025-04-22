from torch import nn
import timm

class EfficientNetB0(nn.Module):
    def __init__(self,num_classes=2):
        super(EfficientNetB0, self).__init__()
        self.model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=num_classes)
    def forward(self, x):
        return self.model(x)
class EfficientNetB1(nn.Module):
    def __init__(self,num_classes=2):
        super(EfficientNetB1, self).__init__()
        self.model = timm.create_model('efficientnet_b1', pretrained=False, num_classes=num_classes)
    def forward(self, x):
        return self.model(x)

class EfficientNetB2(nn.Module):
    def __init__(self,num_classes=2):
        super(EfficientNetB2, self).__init__()
        self.model = timm.create_model('efficientnet_b2', pretrained=False, num_classes=num_classes)
    def forward(self, x):
        return self.model(x)
class EfficientNetB3(nn.Module):
    def __init__(self,num_classes=2):
        super(EfficientNetB3, self).__init__()
        self.model = timm.create_model('efficientnet_b3', pretrained=False, num_classes=num_classes)
    def forward(self, x):
        return self.model(x)