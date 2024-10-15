from config import Config
import matplotlib.pyplot as plt
import torch
import os
# Load global parameters
config = Config()

# Load Config
splits = config.splits
partitions = [0.1,0.2,0.3,0.4,0.5,0.6,0.7]

# File name
name = 'baseline.pt'
baseline = torch.load(f=name)
# Extract indices and baseline jaccard and dice scores
all_indices = baseline['fold_dict']
baseline_jaccard_score = baseline['baseline_jaccard']
baseline_dice_score = baseline['baseline_dice']
baseline_iterations_object = baseline['baseline_iterations']

# Makes directory for models to be saved in
os.makedirs('splits', exist_ok=True)

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
    
    # Save the plot
    plt.tight_layout()
    filename = f'splits/split_{splits[i]}_chart.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)  # Close the figure to free up memory
    
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
    
# Cycle through all splits
plt.rcParams.update({'font.size': 30})
for i in range(len(splits)):
    displayChart(OG_dice[i], new_dice[i], baseline_dice, threshold_dice, OG_jaccard[i], new_jaccard[i], baseline_jaccard, threshold_jaccard, splits, partitions)