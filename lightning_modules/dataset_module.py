import pickle
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms, datasets
import logging
import os
import lightning as L
import glob

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enable debug mode if needed
debug_mode = False
if debug_mode:
    logger.setLevel(logging.DEBUG)

modules_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
current_dir = os.path.dirname(__file__)
rel_path = os.path.relpath(modules_root, start=current_dir)


class ImageTransform():
    def __init__(self, image_size, mean =(0.485, 0.456, 0.406), std =  (0.229, 0.224, 0.225)):
        self.data_transform = {
            'Train': transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(degrees=(-20, 20)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]),
            'Val':  transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)]),
            'Test':  transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)])
        }

    def __call__(self, img, phase='Train'):
        return self.data_transform[phase](img)

class CRDDC_dataset(Dataset):
    def __init__(self, transform=None, phase="Train"):
        self.data_path = "/home/jupyter-iec_thicao/Dataset/RDD2022"
        self.transform = transform
        self.phase = phase
        self.file_list = self.make_datapath_list(phase)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        try:
            img = Image.open(img_path)
            img_transformed = self.transform(img, self.phase)
            parent_dir = img_path.split('/')[-2]
            if parent_dir=="BadRoad":
                label=0
            elif parent_dir=="GoodRoad":
                label=1
            else: label=-1
            return img_transformed, label
        except  Exception as e:
            print(e,img_path)

    def make_datapath_list(self,phase="Train"):
        target_path = os.path.join(self.data_path, phase, "*/*.jpg")
        path_list = []
        for path in glob.glob(target_path):
            path_list.append(path)
        return path_list

class SVRDD_dataset(Dataset):
    def __init__(self, transform=None, phase="Train"):
        self.data_path = "/home/jupyter-iec_thicao/Dataset/SVRDD"
        self.transform = transform
        self.phase = phase
        self.file_list = self.make_datapath_list(phase)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        try:
            img = Image.open(img_path)
            img_transformed = self.transform(img, self.phase)
            parent_dir = img_path.split('/')[-2]
            map_label={
                'Alligator Crack':0,
                'Longitudinal Crack':1,
                'Longitudinal Patch':2,
                'Manhole Cover':3,
                'Pothole':4,
                'Transverse Crack':5,
                'Transverse Patch':6
            }
            label=map_label[parent_dir]
            return img_transformed, label
        except  Exception as e:
            print(e,img_path)

    def make_datapath_list(self,phase="Train"):
        target_path = os.path.join(self.data_path, phase, "*/*.jpg")
        path_list = []
        for path in glob.glob(target_path):
            path_list.append(path)
        return path_list
class RTK_dataset(Dataset):
    def __init__(self, transform=None, phase="Train"):
        self.data_path = "/home/jupyter-iec_thicao/Dataset/RTK"
        self.transform = transform
        self.phase = phase
        self.file_list = self.make_datapath_list(phase)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        try:
            img = Image.open(img_path)
            img_transformed = self.transform(img, self.phase)
            parent_dir = img_path.split('/')[-2]
            if parent_dir=="Asphalt":
                label=0
            elif parent_dir=="Paved":
                label=1
            elif parent_dir=="Unpaved":
                label=2
            else: label=-1
            return img_transformed, label
        except  Exception as e:
            print(e,img_path)

    def make_datapath_list(self,phase="Train"):
        target_path = os.path.join(self.data_path, phase, "*/*.jpg")
        path_list = []
        for path in glob.glob(target_path):
            path_list.append(path)
        return path_list


class Kerusakan_Jalan_dataset(Dataset):
    def __init__(self, transform=None, phase="Train"):
        self.data_path = "/home/jupyter-iec_thicao/Dataset/Kerusakan_Jalan"
        self.transform = transform
        self.phase = phase
        self.file_list = self.make_datapath_list(phase)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        try:
            img = Image.open(img_path)
            img_transformed = self.transform(img, self.phase)
            parent_dir = img_path.split('/')[-2]
            map_label={
                'Normal':0,
                'Crocodile':1,
                'Hole':2,
                'Longitudinal':3,
                'Transverse':4,
            }
            label=map_label[parent_dir]
            return img_transformed, label
        except  Exception as e:
            print(e,img_path)

    def make_datapath_list(self,phase="Train"):
        target_path = os.path.join(self.data_path, phase, "*/*.jpg")
        path_list = []
        for path in glob.glob(target_path):
            path_list.append(path)
        return path_list

class Road_CLS_Quality_dataset(Dataset):
    def __init__(self, transform=None, phase="Train"):
        self.data_path = "/home/jupyter-iec_thicao/Dataset/Road_CLS_Quality"
        self.transform = transform
        self.phase = phase
        self.file_list = self.make_datapath_list(phase)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        try:
            img_path = self.file_list[idx]
            img = Image.open(img_path)
            img_transformed = self.transform(img, self.phase)
            parent_dir = img_path.split('/')[-2]
            map_label={
                'Asphalt_Bad':0,
                'Asphalt_Regular':1,
                'Paved_Bad':2,
                'Paved_Regular':3,
                'Rain':4,
                'Unpaved_Bad':5,
                "Unpaved_Regular":6
            }
            label=map_label[parent_dir]
            return img_transformed, label
        except  Exception as e:
            print("#######################error################")
            print(e,img_path)

    def make_datapath_list(self,phase="Train"):
        target_path = os.path.join(self.data_path, phase, "*/*.jpg")
        path_list = []
        for path in glob.glob(target_path):
            path_list.append(path)
        return path_list




class DatasetModule(L.LightningDataModule):
    def __init__(self, dataset_name,batch_size=64, image_size=224, num_workers=15):
        super().__init__()
        self.val_dataset = None
        self.train_dataset = None
        self.test_dataset = None
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size

    def setup(self, stage: str):
        if self.dataset_name == "CRDDC":
            self.train_dataset= CRDDC_dataset(transform=ImageTransform(self.image_size), phase="Train")
            self.test_dataset= CRDDC_dataset(transform=ImageTransform(self.image_size), phase="Test")
            self.val_dataset=self.test_dataset
        elif self.dataset_name == "SVRDD":
            self.train_dataset= SVRDD_dataset(transform=ImageTransform(self.image_size), phase="Train")
            self.test_dataset= SVRDD_dataset(transform=ImageTransform(self.image_size), phase="Test")
            self.val_dataset=self.test_dataset
        elif self.dataset_name == "RTK":
            self.train_dataset = RTK_dataset(transform=ImageTransform(self.image_size), phase="Train")
            self.test_dataset = RTK_dataset(transform=ImageTransform(self.image_size), phase="Test")
            self.val_dataset=self.test_dataset
        elif self.dataset_name == "KJ":
            self.train_dataset=Kerusakan_Jalan_dataset(transform=ImageTransform(self.image_size), phase="Train")
            self.test_dataset=Kerusakan_Jalan_dataset(transform=ImageTransform(self.image_size), phase="Test")
            self.val_dataset=self.test_dataset
        elif self.dataset_name == "Road_CLS_Quality":
            self.train_dataset=Road_CLS_Quality_dataset(transform=ImageTransform(self.image_size), phase="Train")
            self.val_dataset=Road_CLS_Quality_dataset(transform=ImageTransform(self.image_size), phase="Val")
            self.test_dataset=Road_CLS_Quality_dataset(transform=ImageTransform(self.image_size), phase="Test")
        else: raise ValueError("Invalid dataset name")


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True,num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, drop_last=True, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, drop_last=True, num_workers=self.num_workers)