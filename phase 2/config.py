import torch
from torchmetrics import JaccardIndex
from torchmetrics import Dice


# Dataset import
from dataset import ISICSegmentationDataset

class Config():
    def __init__(self):
        # Dataset config
        self.input_size = 256
        self.image_path = "./ISIC/images/ISIC2018_Task1-2_Training_Input/"
        self.mask_path = "./ISIC/masks/ISIC2018_Task1_Training_GroundTruth/"
        self.splits = ['A','B','C','D','E']
        self.partitions = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7]
        # Hyperparameters
        self.partitions = 5
        self.BATCH_SIZE = 16
        self.num_workers = 8
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Loss related
        self.loss_fn = torch.nn.BCELoss()
        self.threshold = 0.001
        self.alpha = 0.9
        self.lr = 0.01
        # Measurement
        self.jaccard = JaccardIndex(task='multiclass', num_classes = 2, average = 'micro').to(self.device)
        self.dice = Dice(num_classes = 2, average = 'micro').to(self.device)
        # Dataset
        dataset = ISICSegmentationDataset(self.image_path, self.mask_path, self.image_size)
        