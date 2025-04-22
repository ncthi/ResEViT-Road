import timm
from torch import nn

class ResNet18(nn.Module):
    def __init__(self,num_classes=2):
        super(ResNet18, self).__init__()
        self.model = timm.create_model('resnet18', pretrained=False, num_classes=num_classes)
    def forward(self, x):
        return self.model(x)
class ResNet50(nn.Module):
    def __init__(self,num_classes=2):
        super(ResNet50, self).__init__()
        self.model = timm.create_model('resnet50', pretrained=False, num_classes=num_classes)
    def forward(self, x):
        return self.model(x)
class ResNet34(nn.Module):
    def __init__(self,num_classes=2):
        super(ResNet34, self).__init__()
        self.model = timm.create_model('resnet34', pretrained=False, num_classes=num_classes)
    def forward(self, x):
        return self.model(x)