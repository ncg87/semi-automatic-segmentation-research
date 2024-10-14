# General use for DL
import torch
import numpy as np
import matplotlib.pyplot as plt

# Used to split data
from sklearn.model_selection import KFold

# To loader dataset
from torch.utils.data import DataLoader

# Data Measurements
from torchmetrics import JaccardIndex
from torchmetrics import Dice

# Useful functions
from exp_utils import *
from train import *

from base_dataset import BaseDataset

## Preliminary variables ##

# Specifies whether to train on GPU or CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Loss for training
loss_fn = torch.nn.BCELoss()

# Measurements
jaccard = JaccardIndex(task='multiclass', num_classes = 2, average = 'micro').to(device)
dice = Dice(num_classes = 2, average = 'micro').to(device)

# K-Fold object for splitting dataset, randomizes batches (shuffle = True)
splits = 5
kf = KFold(splits, shuffle=True)

# Batch Size
BATCH_SIZE = 16

# Stopping threshold
threshold = 0.001

# Speeds up training
num_workers = 8

# Alpha for EMA
alpha = 0.9








# Create the baseline
# Dictionary to save each fold
splits = ['A','B','C','D','E'] # To save indicies of each fold
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
    lr = 0.01
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


# File name
name = 'baseline.pt'
baseline = torch.load(f=name)
# Extract indices and baseline jaccard and dice scores
all_indices = baseline['fold_dict']
baseline_jaccard = baseline['baseline_jaccard']
baseline_dice = baseline['baseline_dice']


partitions = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7]
# Cycle through all test splits
for split in splits:
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
        lr = 0.01
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
        lr = 0.01
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
        
        
## Baseline Data ##

# File name
name = 'baseline.pt'
baseline = torch.load(f=name)
# Extract indices and baseline jaccard and dice scores
all_indices = baseline['fold_dict']
baseline_jaccard_score = baseline['baseline_jaccard']
baseline_dice_score = baseline['baseline_dice']
baseline_iterations_object = baseline['baseline_iterations']


splits = ['A', 'B', 'C', 'D', 'E']
partitions = [0.1,0.2,0.3,0.4,0.5,0.6,0.7]

def displayChart(OG_dice, new_dice, baseline_dice, threshold_dice, OG_jaccard, new_jaccard, baseline_jaccard, threshold_jaccard, splits, partitions):
    fig, ax = plt.subplots(2, 1, figsize=(12, 30))
    ax[0].plot(range(len(partitions)), OG_jaccard, label='Original Jaccard', marker='o', markersize=20)
    ax[0].plot(range(len(partitions)), new_jaccard, label='Jaccard after SAS', marker='x', markersize=20)
    ax[0].plot(range(len(partitions)), baseline_jaccard, label='Baseline Jaccard')
    ax[0].plot(range(len(partitions)), threshold_jaccard, label='90% Threshold', linestyle='--')
    ax[0].set_title(f'Split: {splits[i]} Jaccard Scores')
    ax[0].set_xlabel('Data Partition Size')
    ax[0].set_ylabel('Jaccard Score')
    ax[0].set_xticks(range(len(partitions)))
    ax[0].set_xticklabels(partitions)
    ax[0].set_ylim([0, 1])
    ax[0].legend()
    
    # Scatter Dice scores
    ax[1].plot(range(len(partitions)), OG_dice, label='Original Dice', marker='o', markersize=20)
    ax[1].plot(range(len(partitions)), new_dice, label='Dice after SAS', marker='x', markersize=20)
    ax[1].plot(range(len(partitions)), baseline_dice, label='Baseline Dice')
    ax[1].plot(range(len(partitions)), threshold_dice, label='90% Threshold', linestyle='--')
    ax[1].set_title(f'Split: {splits[i]} Dice Scores')
    ax[1].set_xlabel('Data Partition Size')
    ax[1].set_ylabel('Dice Score')
    ax[1].set_xticks(range(len(partitions)))
    ax[1].set_xticklabels(partitions)
    ax[1].set_ylim([0, 1])
    ax[1].legend()
    plt.pause(0.001)
    
# Initalize lists to store data
baseline_jaccard = []
threshold_jaccard = []
OG_jaccard = []
new_jaccard = []

baseline_dice = []
threshold_dice = []
OG_dice = []
new_dice = []

baseline_iterations = []
OG_iterations = []
new_iterations = []

# Fill baseline list
for parition in partitions:
    baseline_jaccard.append(baseline_jaccard_score)
    baseline_dice.append(baseline_dice_score)
    threshold_jaccard.append(baseline_jaccard_score * 0.9)
    threshold_dice.append(baseline_dice_score * 0.9)
    baseline_iterations.append(baseline_iterations_object)

# Cycle through all models and data points for each
for split in splits:
    # Initialize the arrays
    split_OG_jaccard = []
    split_OG_dice = []
    split_OG_iterations = []
    
    split_new_jaccard = []
    split_new_dice = []
    split_new_iterations = []
    
    for partition in partitions:
        # Load the stored data
        name = f'./new_models/Split:{split}|Partition:{partition}|New'
        data = torch.load(name)
        # Extract data
        split_new_jaccard.append(data['jaccard_score'])
        split_new_dice.append(data['dice_score'])
        split_new_iterations.append(data['num_iterations'])
        split_OG_jaccard.append( data['orig_jaccard_score'])
        split_OG_dice.append(data['orig_dice_score'])
        split_OG_iterations.append(data['orig_iterations'])
    # Store arrays for plotting
    OG_jaccard.append(split_OG_jaccard)
    new_jaccard.append(split_new_jaccard)
    
    OG_dice.append(split_OG_dice)
    new_dice.append(split_new_dice)
    
    new_iterations.append(split_new_iterations)
    OG_iterations.append(split_OG_iterations)
    

# Cycle through all splits
plt.rcParams.update({'font.size': 30})
for i in range(len(splits)):
    displayChart(OG_dice[i], new_dice[i], baseline_dice, threshold_dice, OG_jaccard[i], new_jaccard[i], baseline_jaccard, threshold_jaccard, splits, partitions)

    


    
        
        