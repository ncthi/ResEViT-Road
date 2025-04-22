import timm
from torch import nn

class Inception_v4(nn.Module):
    def __init__(self, num_classes=2):
        super(Inception_v4, self).__init__()
        self.nodel=timm.create_model("inception_v4", pretrained=False, num_classes=num_classes)
    def forward (self,x):
        x = self.nodel(x)
        return x