
import random

import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as F
from tqdm import tqdm
from skimage.transform import resize
from skimage import io, color


class PolypsSegmentationDataset(Dataset):
    '''KVASIR Colon Polyps dataset'''
    
    # -- Initializes dataset
    def __init__(self, root_path, image_size=256, augumented = False):
        # define root path to images
        self.root_path = root_path
        self.image_size = image_size
        self.augumented = augumented
        # read paths to all images
        self.images = sorted([os.path.join(root_path,"images",i) for i in tqdm(os.listdir(os.path.join(root_path,"images")))])
        self.masks = sorted([os.path.join(root_path,"masks",i) for i in tqdm(os.listdir(os.path.join(root_path,"masks")))])
        
        # defining the transform to images
        self.transform = transforms.Compose(
            [transforms.Resize(image_size),
            transforms.ToTensor()]
        )
    
    # -- Reshapes and reformats ski-kit image to pytorch image format
    def reshape(self, img):
        # store each channel
        reshaping_list = []
        for i in range(img.shape[2]):
            reshaping_list.append(torch.unsqueeze(torch.from_numpy(img[:,:,i]),0))
        # concatenate over 0th dimension
        new_img = torch.concat(reshaping_list, 0)
        
        return new_img    
    
    # -- Returns image and mask at a certain index
    def __getitem__(self, idx):
        
        img = io.imread(self.images[idx])
        mask = io.imread(self.masks[idx])
        
        #scikit image rescaling
        img_rescaled = resize(img, (self.image_size, self.image_size))
        mask_rescaled = color.rgb2gray(resize(mask, (self.image_size, self.image_size)))
        
        true_img = self.reshape(img_rescaled).to(torch.float32)
        true_mask = torch.unsqueeze(torch.from_numpy(mask_rescaled),0).to(torch.float32)
        
        if not self.augumented:
            return true_img, true_mask
        
        # random chance of horizontal flip
        if(random.random() < 0.5):
            true_img = F.hflip(true_img)
            true_mask = F.hflip(true_mask)
        # random chance of vertical flip
        if(random.random() < 0.5):
            true_img = F.vflip(true_img)
            true_mask = F.vflip(true_mask)
        # random rotation
        rotation = random.random() * 360
        true_img = F.rotate(true_img, rotation)
        true_mask = F.rotate(true_mask, rotation)
        
        return true_img, true_mask
    # -- Returns length of dataset
    def __len__(self):
        return len(self.images)
    
        
        
