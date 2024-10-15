# General use for DL
import torch
import os
# To loader dataset
from torch.utils.data import DataLoader
# Useful functions
from exp_utils import *
from train import *

from config import Config
from base_dataset import BaseDataset

# Load global parameters
config = Config()

## Preliminary variables ##

# Specifies whether to train on GPU or CPU
device = config.device
# Loss for training
loss_fn = config.loss_fn

# Measurements
jaccard = config.jaccard
dice = config.dice

# Batch Size
BATCH_SIZE = config.BATCH_SIZE

# Stopping threshold
threshold = config.threshold

# Speeds up training
num_workers = config.num_workers

# Alpha for EMA
alpha = config.alpha

# Define dataset
dataset = config.dataset

# Load Config
splits = config.splits
partitions = config.partitions

# Makes directory for new models to be saved in
os.makedirs('new_models', exist_ok=True)

# Cycle through all models
for split in splits:
    for partition in partitions:
        # Load the stored data
        name = f'./model/Split:{split}|Partition:{partition}|New'
        data = torch.load(name)
        # Extract data
        train_indices = data['train_indices']
        remaining_indices = data['remaining_indices']
        test_indices = data['test_indices']
        orig_jaccard_score = data['jaccard_score']
        orig_dice_score = data['dice_score']
        orig_iterations = data['num_iterations']
        state_dict = data['state_dict']
        # Load base model
        trained_model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
            in_channels=3, out_channels=1, init_features=64, pretrained=False, trust_repo=True).to(device)
        # Load saved model
        trained_model.load_state_dict(state_dict)
        
        # Create new dataset
        new_dataset = []
        # Get train loader for fold
        remaining_loader = DataLoader(
            dataset=dataset,
            batch_size=1,
            sampler=torch.utils.data.SubsetRandomSampler(remaining_indices),
            num_workers = num_workers
        )
        # Create masks of remaining data
        images, new_masks = create_masks(trained_model, device, remaining_loader)
        # Create dataset with new masks
        new_dataset.append(BaseDataset(images, new_masks))
        # Get images and masks used to train saved model
        base_images, base_masks = get_masks(train_indices,dataset)
        # Create dataset with ground truth masks and images
        new_dataset.append(BaseDataset(base_images, base_masks))
        # Concatenate the two so we have a dataset with generated masks and truth maks
        # this will be our train dataset
        train_dataset = torch.utils.data.ConcatDataset(new_dataset)
        
        # Create train loader for new dataset
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size = BATCH_SIZE,
            shuffle = True,
            num_workers = num_workers,
        )    
        
        ## Initialize the model ##
        
        # We will begin our learning rate at 0.01 
        lr = config.lr
        # Optimizer for model
        optimizer = torch.optim.Adam(trained_model.parameters(), lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,100)
        
        ## Initialize the training ##
    
        # Initialize previous and current loss for stopping rule
        
        L_MA = 1 # Moving average of loss
        L_k = 0 # Current loss
        
        # To determine in threshold is too low
        counter = 0
        # Train model until stopping rule is reached
        while(stopping_rule(L_MA, L_k, threshold) or counter < 10):
            # Train model and compute loss
            L_k = train_model(trained_model, loss_fn, device, train_loader, optimizer)
            
            # Initialization
            if(L_MA == 0):
                L_MA = L_k
            
            # Find EMA of losses
            L_MA = moving_avg(alpha, L_MA, L_k)
            counter += 1
            
        # To determine in threshold is too low
        print(counter) 
        
        # Get test dataset
        test_loader = DataLoader(
                dataset=dataset,
                batch_size=BATCH_SIZE,
                sampler=torch.utils.data.SubsetRandomSampler(test_indices),
                num_workers = num_workers,
        )
        
        # Test model on test split
        jaccard_score, dice_score = test_model(trained_model, device, test_loader, jaccard, dice)
        print(f'Split:{split}|Partition:{partition}|NewJaccard:{jaccard_score}|OGJaccard:{orig_jaccard_score}|NewDice:{dice_score}|OGDice:{orig_dice_score}')
        name = f'./new_models/Split:{split}|Partition:{partition}|New'
        # Save model and results
        state = {
            'state_dict' : trained_model.state_dict(),
            'jaccard_score' : jaccard_score,
            'dice_score' : dice_score,
            'num_iterations' : counter,
            'orig_jaccard_score' : orig_jaccard_score,
            'orig_dice_score' : orig_dice_score,
            'orig_iterations' : orig_iterations,
        }
        torch.save(state,f=name)