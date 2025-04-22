from lightning_modules import DatasetModule,MyModel
from sklearn.metrics import classification_report
import numpy as np
import os
import glob
import torch

class Predictor:
    def __call__(self,output):
        output = output.cpu()
        max_id = np.argmax(output.detach().numpy(),axis=1)
        return max_id

class Class_report:
    def __init__(self,data,model,device):
        self.data=data
        self.model=model
        self.device=device
    def __call__(self):
        self.model.eval()
        predicts=[]
        label_co=[]
        for inputs, labels in self.data:
            inputs=inputs.to(self.device)
            output = self.model(inputs)
            label_co=np.concatenate((label_co,labels))
            response = Predictor()(output)
            predicts=np.concatenate((predicts,response))
        print(classification_report(label_co,  predicts,digits=4))

class ListModel_report:
    def __init__(self,model_list:list,dataset_name="CRDDC",device="cuda"):
        self.model_list=model_list
        self.dataset_name=dataset_name
        mapping_num_classes={
            "CRDDC":2,
            "SVRDD":7,
            "RTK":3,
            "KJ":5,
            "Road_CLS_Quality":7

        }
        self.num_classes=mapping_num_classes[self.dataset_name]
        self.device=device
    def __call__(self):
        for model_name,model_size in self.model_list:
            print(f"Model:{model_name} Size:{model_size}")
            model=MyModel(model_name,model_size,num_classes=self.num_classes)
            checkpoint=self.get_latest_checkpoint(model_name,model_size,self.dataset_name)
            model.load_state_dict(torch.load(checkpoint))
            model=model.to(self.device)
            data=DatasetModule(self.dataset_name,batch_size=16)
            data.setup(stage="test")
            Class_report(data.test_dataloader(),model,self.device)()
    @staticmethod
    def get_latest_checkpoint(model_name: str, model_size: str,dataset:str):
        checkpoint_list = glob.glob(f'/home/jupyter-iec_thicao/ResEViT-Road/models/{dataset}/{model_name}_{model_size}*state_dict.pt')

        if not checkpoint_list:
            raise FileNotFoundError(f"No checkpoint found for model name: {model_name}, model size: {model_size}")
        latest_ckpt = max(checkpoint_list, key=os.path.getctime)
        return latest_ckpt
