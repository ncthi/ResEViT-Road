from torch import nn
import timm

class MobileViT_xs(nn.Module):
    def __init__(self,num_classes=2):
        super(MobileViT_xs, self).__init__()
        self.model = timm.create_model('mobilevit_xs', pretrained=False, num_classes=num_classes)
    def forward(self, x):
        return self.model(x)
class MobileViT_s(nn.Module):
    def __init__(self,num_classes=2):
        super(MobileViT_s, self).__init__()
        self.model = timm.create_model('mobilevit_s', pretrained=False, num_classes=num_classes)
    def forward(self,x):
        return self.model(x)
class MobileViT_xxs(nn.Module):
    def __init__(self,num_classes=2):
        super(MobileViT_xxs, self).__init__()
        self.model = timm.create_model('mobilevit_xxs', pretrained=False, num_classes=num_classes)
    def forward(self,x):
        return self.model(x)