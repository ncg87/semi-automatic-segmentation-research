import traceback

# General use for DL
import torch
import numpy as np

# To loader dataset
from torch.utils.data import DataLoader

# Data Measurements
from torchmetrics import JaccardIndex
from torchmetrics import Dice

# Dataset import
from dataset import Kvasir
from transformers import SegformerImageProcessor, MaskFormerImageProcessor, AutoImageProcessor

# For sending an email
import utils
import dotenv 

env_vars = dotenv.dotenv_values(dotenv_path="./.env")

from exp_utils import stopping_rule, moving_avg
from train import train_model, test_model
from utils import send_notification
# Paths to masks and images
image_path = "./data/Kvasir-SEG/images"
masks_path = "./data/Kvasir-SEG/masks/"  
# Size of image
size = 256
# Define dataset
processor = SegformerImageProcessor.from_pretrained(env_vars["pretrained_variant"])
dataset = Kvasir(dataset_path=f'{env_vars["data_folder_path"]}/{env_vars["dataset_name"]}',
                     processor=processor,
                     image_size=(256,256),
                     mask_size=(256, 256),
                     )

## Preliminary variables ##

# Specifies whether to train on GPU or CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Loss for training
loss_fn = torch.nn.BCELoss()

# Measurements
jaccard = JaccardIndex(task='multiclass', num_classes = 2, average = 'micro').to(device)
dice = Dice(num_classes = 2, average = 'micro').to(device)

# Batch Size
BATCH_SIZE = 16

# Stopping threshold
threshold = 0.001

# Speeds up training
num_workers = 8

# Alpha for EMA
alpha = 0.9

# File name
name = 'baseline.pt'
baseline = torch.load(f=name)
# Extract indices and baseline jaccard and dice scores
all_indices = baseline['fold_dict']
if __name__ == "__main__":
    try:
        splits = ['A', 'B', 'C', 'D', 'E']
        split_dices = {}
        for split in splits:
            # Get indices of test and train points in dataset
            indices = all_indices[split]
            train_indices = indices[0]
            test_indices = indices[1]
            
            # Array to store each splits dices
            test_dice = []
            
            # Create test dataloader for fold
            test_loader = DataLoader(
                dataset=dataset,
                batch_size=BATCH_SIZE,
                sampler=torch.utils.data.SubsetRandomSampler(test_indices),
                num_workers = num_workers
            )
            
            # Array to store indices of model, increased by 10 at a time until we get to the whole train set
            model_indices = []
            
            while len(train_indices) > 0:
                # To prevent selection of more points than available, causing error
                n_to_select = min(10, len(train_indices))
                # Get a random 10 point of the train set
                model_indices.extend(np.random.choice(train_indices, n_to_select, replace=False))
                # Remove the indices from the the next round of selection
                train_indices = np.setdiff1d(train_indices, model_indices) 
                # Create a train dataloader for partition
                train_loader = DataLoader(
                    dataset=dataset,
                    batch_size = BATCH_SIZE,
                    sampler=torch.utils.data.SubsetRandomSampler(model_indices),
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
                
                L_MA = 1 # Moving average of loss, intialized to zero for error purposes
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
                # Get the dice score of the trained model
                jaccard_score, dice_score = test_model(model, device, test_loader, jaccard, dice)
                print(f"Model split {split}, {len(model_indices)} data points | Dice Score: {dice_score}")
                # Store the dice score
                test_dice.append(dice_score)
                
            # Store that splits dice scores in dictionary
            split_dices[split] = test_dice

        # Store the dictionary of dice scores
        torch.save(split_dices, 'split_dices.pt')
        response = send_notification(subject="Expectations Finished",
                                        body="Expectations Finished on billbergia",
                                        sender_email = str(env_vars["sender_email"]),
                                        receiver_email = env_vars["receiver_email"],
                                        smtp_server = str(env_vars["smtp_server"]),
                                        smtp_port = int(env_vars["smtp_port"]),
                                        password = str(env_vars["password"]),
                                        results_fname="split_dices.pt",
                                        results_fpath=f'./split_dices.pt')
    except Exception as e:
        traceback.print_exc()
        send_notification(subject="Expectations Error",
                          body=f"Error in expectations: {e}",
                          sender_email=env_vars["sender_email"],
                          receiver_email=env_vars["receiver_email"],
                          smtp_server=str(env_vars["smtp_server"]),
                          smtp_port=int(env_vars["smtp_port"]),
                          password=str(env_vars["password"]))

