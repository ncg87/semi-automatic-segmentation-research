# General use for DL
import torch
# To load datasets
from torch.utils.data import DataLoader
# Used to split data
from sklearn.model_selection import KFold
# Useful functions
from exp_utils import *
from train import *
# Contains all global parameters
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

# K-Fold object for splitting dataset, randomizes batches (shuffle = True)
splits = config.partitions
kf = KFold(splits, shuffle=True)

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

######## ----- Run the Experiment  ----- ########

# Create the baseline
# Dictionary to save each fold
splits = config.splits # To save indicies of each fold
fold_dict = {}
# Initialize Jaccard and Dice
average_jaccard = 0
average_dice = 0
average_iterations = 0

# Split dataset and determine baseline jaccard and dice
for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):
    
    ## Initialize the datasets ##
    
    # Get train loader for fold
    train_loader = DataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        sampler=torch.utils.data.SubsetRandomSampler(train_idx),
        num_workers = num_workers
    )
    # Get test loader for fold
    test_loader = DataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        sampler=torch.utils.data.SubsetRandomSampler(test_idx),
        num_workers = num_workers
    )
    
    # Save indices in dictionary for future experiments
    fold_dict.update({splits[fold] : [train_idx, test_idx]})
    
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

    L_MA = 1 # Moving average of loss, initialize to 0 so no division by zero error
    L_k = 0 # Current loss

    # To determine in threshold is too low
    counter = 0
    # Train model until stopping rule is reached
    while(stopping_rule(L_MA, L_k, threshold) or counter < 10):
        # Train model and compute loss
        L_k = train_model(model, loss_fn, device, train_loader, optimizer)

        # Initialization
        if(L_MA == 0):
            L_MA = L_k

        # Find EMA of losses
        L_MA = moving_avg(alpha, L_MA, L_k)
        counter += 1
    # To determine in threshold is too low
    print(f"Iterations:{counter} | Loss: {L_k}")
    # Test model on remaining splits
    jaccard_i, dice_i = test_model(model, device, test_loader, jaccard, dice)
    average_jaccard += jaccard_i
    average_dice += dice_i
    average_iterations += counter

# Take average of jaccard and dice of all 5 models
average_jaccard /= 5    
average_dice /= 5
average_iterations /= 5

# Print results
print(f"Baseline Jaccard: {average_jaccard}")
print(f"Baseline Dice: {average_dice}")

# Save the indicies and baseline scores into a state dict for future use
name = 'baseline.pt'
state = {
    'fold_dict' : fold_dict,
    'baseline_jaccard' : average_jaccard,
    'baseline_dice' : average_dice,
    'baseline_iterations' : average_iterations
}
torch.save(state, f = name)
