# General use for DL
import torch
import numpy as np
import os
# To loader dataset
from torch.utils.data import DataLoader
# Useful functions
from exp_utils import *
from train import *
from config import Config





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

# File name
name = 'baseline.pt'
baseline = torch.load(f=name)
# Extract indices and baseline jaccard and dice scores
all_indices = baseline['fold_dict']
baseline_jaccard = baseline['baseline_jaccard']
baseline_dice = baseline['baseline_dice']

# Load Config
splits = config.splits
partitions = config.partitions

# Makes directory for models to be saved in
os.makedirs('model', exist_ok=True)

# Cycle through all test splits
for split in config:
    # Get indices of test and train points in dataset
    indices = all_indices[split]
    train_indices = indices[0]
    test_indices = indices[1]
    # Split train indices into eighths
    partition_indices = split_into_eights(train_indices)
    # Create test dataloader for fold
    test_loader = DataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        sampler=torch.utils.data.SubsetRandomSampler(test_indices),
        num_workers = num_workers
    )
    # Cycle through all partition lengths of training data (0.1,0.2,...,0.7)
    for i in range(1, len(partition_indices)):
        
        # Initialize Jaccard and Dice
        average_jaccard = 0
        average_dice = 0
                
        train_indices_i = np.hstack(partition_indices[0:i])
        remaining_indices = np.hstack(partition_indices[i:])
        
        # Create a train dataloader for partition
        train_loader = DataLoader(
            dataset=dataset,
            batch_size = BATCH_SIZE,
            sampler=torch.utils.data.SubsetRandomSampler(train_indices_i),
            num_workers = num_workers
        )
        
        ## Initialize the model ##

        # Loading an untrained model to GPU/CPU
        model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
            in_channels=3, out_channels=1, init_features=64, pretrained=False, trust_repo=True).to(device)
        # We will begin our learning rate at 0.01 
        lr = config.lr
        # Optimizer for model
        optimizer = torch.optim.Adam(model.parameters(), lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,25)
        
        ## Initialize the training ##
    
        # Initialize previous and current loss for stopping rule
        
        L_MA = 1 # Moving average of loss
        L_k = 0 # Current loss
        
        # To determine in threshold is too low
        counter = 0
        # Train model until stopping rule is reached
        while(stopping_rule(L_MA, L_k, threshold) or counter < 10):
            # Train model and compute loss
            L_k = train_model(model, loss_fn, device, train_loader, optimizer)
            
            # Initialization of EMA
            if(L_MA == 0):
                L_MA = L_k
            
            # Find EMA of losses
            L_MA = moving_avg(alpha, L_MA, L_k)
            counter += 1
               
        # To determine in threshold is too low
        print(counter)
        # Test model on test split
        jaccard_score, dice_score = test_model(model, device, test_loader, jaccard, dice)
        print(f'Split:{split}|Partition:{partitions[i]}|Jaccard:{jaccard_score}|Dice:{dice_score}')
        # Save model, scores, and indices for next step
        name = f'./model/Split:{split}|Partition:{partitions[i]}|New'
        state = {
            'train_indices' : train_indices_i,
            'remaining_indices' : remaining_indices,
            'test_indices' : test_indices,
            'jaccard_score' : jaccard_score,
            'dice_score' : dice_score,
            'num_iterations' : counter,
            'state_dict' : model.state_dict(),
        }
        torch.save(state, f=name)
