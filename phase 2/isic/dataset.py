import random
import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as F
from tqdm import tqdm
from skimage.transform import resize
from skimage import io, color

class ISICSegmentationDataset(Dataset):
    """ISIC Challenge dataset"""
    def __init__(self, image_path, mask_path, image_size = 256, augumented = False):
        # Define paths to images and masks
        self.image_path = image_path
        self.mask_path = mask_path
        self.augumented = augumented
        self.image_size = image_size
        # Read paths to all images
        self.images = sorted(tqdm(glob.glob(os.path.join(image_path,"*.jpg"))))
        # Read path to all masks
        self.masks = sorted(tqdm(glob.glob(os.path.join(mask_path,"*.png"))))
        
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
        
        # Load image and mask using Pillow
        img = Image.open(self.images[idx]).convert('RGB')
        mask = Image.open(self.masks[idx]).convert('L')

        # Resize image and mask
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)

        # Convert to PyTorch tensors
        img_tensor = F.to_tensor(img).to(torch.float32)
        mask_tensor = F.to_tensor(mask).to(torch.float32)

        if not self.augumented:
            return img_tensor, mask_tensor
        
        # random chance of horizontal flip
        if(random.random() < 0.5):
            true_img = F.hflip(img_tensor)
            true_mask = F.hflip(mask_tensor)
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