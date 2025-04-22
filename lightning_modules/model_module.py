import logging
import lightning as L
import torch
import torch.nn as nn
import torchmetrics as tm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Baseline import ResNet18, ResNet50, ResNet34, EfficientViT_B1, EfficientViT_B2, EfficientNetB0, EfficientNetB1, \
    EfficientNetB2, EfficientNetB3, CoaNet_0, RB5DFCNN, CoaNet_1, MobileViT_xs,MobileViT_s,MobileViT_xxs,MobileNet_large, Inception_v4
from ResEViT_Road import ResEViT_road_cls
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)




######### get model ######
def get_model(model_name: str, size: str, **kwargs):
    model_map = {
        "resnet": {
            "18": ResNet18(**kwargs),
            "34": ResNet34(**kwargs),
            "50": ResNet50(**kwargs)
        },
        "efficientvit": {
            "b1": EfficientViT_B1(**kwargs),
            "b2": EfficientViT_B2(**kwargs),
        },
        "efficientnet":{
            "b0": EfficientNetB0(**kwargs),
            "b1": EfficientNetB1(**kwargs),
            "b2": EfficientNetB2(**kwargs),
            "b3": EfficientNetB3(**kwargs),
        },
        "coanet":{
            "0": CoaNet_0(**kwargs),
            "1": CoaNet_1(**kwargs),
        },
        "rd_5dfcnn":{
            "standard":RB5DFCNN(**kwargs)
        },
        "resevit_road":{
            "standard": ResEViT_road_cls(**kwargs),
        },
        "mobilevit":{
            "s":MobileViT_s(**kwargs),
            "xs":MobileViT_xs(**kwargs),
            "xxs":MobileViT_xxs(**kwargs),
        },
        "mobilenet":{
            "large":MobileNet_large(**kwargs),
        },
        "inception":{
            "v4": Inception_v4(**kwargs),
        }
    }

    return model_map[model_name][size]

########################## Lightning Module ##########################

class MyModel(L.LightningModule):
    def __init__(self, model_name, model_size,image_size=224,lr=1e-3,num_classes=2):
        super().__init__()
        self.lr = lr
        self.loss_func = nn.CrossEntropyLoss()
        if num_classes==2: task='binary'
        else: task='multiclass'
        self.accuracy = tm.Accuracy(task=task, num_classes=num_classes)
        self.save_hyperparameters()
        self.model=get_model(model_name,size=model_size,num_classes=num_classes)
        logger.info(f"{model_name}_{model_size}")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, logger=True, prog_bar=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, logger=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, logger=True, prog_bar=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, logger=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)
        self.log('test_loss', loss, on_step=False, on_epoch=True, logger=True, prog_bar=True)
        self.log('test_acc', acc, on_step=False, on_epoch=True, logger=True, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        return y_hat

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, mode='min', patience=5, verbose=True),
            'monitor': 'val_loss'
        }
        return [optimizer], [scheduler]