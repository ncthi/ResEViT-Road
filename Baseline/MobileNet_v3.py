import torch.nn as nn
import timm

class MobileNet_large(nn.Module):
    def __init__(self, num_classes=2):
        super(MobileNet_large, self).__init__()
        self.model=timm.create_model('mobilenetv3_large_075', pretrained=False,num_classes=num_classes)
    def forward(self, x):
        x = self.model(x)
        return x


