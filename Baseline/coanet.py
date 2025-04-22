import timm
from torch import nn

class CoaNet_0(nn.Module):
    def __init__(self,num_classes=2):
        super(CoaNet_0, self).__init__()
        self.model = timm.create_model('coatnet_0_224', pretrained=False, num_classes=num_classes)
    def forward(self, x):
        return self.model(x)

class CoaNet_1(nn.Module):
    def __init__(self,num_classes=2):
        super(CoaNet_1, self).__init__()
        self.model = timm.create_model('coatnet_1_224', pretrained=False, num_classes=num_classes)
    def forward(self, x):
        return self.model(x)