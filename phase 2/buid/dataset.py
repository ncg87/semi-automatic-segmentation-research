import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as F
import os
from PIL import Image

class BUIDSegmentationDataset(Dataset):
    def __init__(self, root_dir, size=256, augment=False):
        self.root_dir = root_dir
        self.size = size
        self.augment = augment
        self.image_files = []
        self.mask_files = []

        for subfolder in os.listdir(root_dir):
            subfolder_path = os.path.join(root_dir, subfolder)
            if os.path.isdir(subfolder_path):
                for file_name in os.listdir(subfolder_path):
                    if file_name.endswith('.png') and '_mask' not in file_name:
                        image_file = os.path.join(subfolder_path, file_name)
                        mask_file = os.path.join(subfolder_path, file_name.replace('.png', '_mask.png'))
                        if os.path.exists(mask_file):
                            self.image_files.append(image_file)
                            self.mask_files.append(mask_file)

        self.transform = transforms.Compose([
            transforms.Resize((self.size, self.size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.augmentation = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
        ])
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_files[idx]).convert('RGB')
        mask = Image.open(self.mask_files[idx]).convert('L')
        
        if self.augment:
            image = self.augmentation(image)
            mask = self.augmentation(mask)
        
        image = self.transform(image)
        mask = transforms.Resize((self.size, self.size))(mask)
        mask = transforms.ToTensor()(mask)
        
        return image, mask