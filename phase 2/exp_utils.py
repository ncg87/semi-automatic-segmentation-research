import torch

# Finds percent change between previous and current loss
# and if it is less than threshold return false
def stopping_rule(L_k, L_k1, threshold):
    return abs(L_k - L_k1) / L_k > threshold

# Calculates the moving average
def moving_avg(alpha, L_MA, L_k):
    return alpha * L_MA + (1-alpha) * L_k

def split_into_eights(list):
    # Floor division of length of list
    partition_size = len(list) // 8
    remainder  = len(list) % 8
    
    # List that will store each parition
    partitions = []
    
    # Partition the list, if partition is not even distrubutes 
    # remainder between beginning paritions
    start = 0
    for i in range(8):
        end = start + partition_size + (1 if i < remainder and i != 0 else 0)
        partitions.append(list[start:end])
        start = end
    return partitions  

def get_masks(indices, dataset):
    masks = []
    images = []
    for i in indices:
        image, mask = dataset.__getitem__(i)
        images.append(image)
        masks.append(mask)
    return images, masks

def create_masks(model, device, loader):
    # Initialize 
    masks = []
    images = []
    # Create masks
    model.eval()
    for batch_idx, (X,y) in enumerate(loader):
        # Get batch
        image, mask = X.to(device), y.to(device)
        # Get results
        output = model(image)
        # Detach from CPU and squeeze batch(1) dimension
        masks.append(output.detach().cpu().squeeze(0))
        images.append(image.detach().cpu().squeeze(0))
    # Segmented masks
    return images, masks
    