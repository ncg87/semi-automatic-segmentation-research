import torch

def train_model(model, loss_fn, device, train_loader, optimizer):
    # Initalize loss
    average_loss = 0
    # Train on dataset
    model.train()
    for batch_idx, (X,y) in enumerate(train_loader):
        # Get batch
        image, mask = X.to(device), y.to(device)
        # Get results
        output = model(image)
        # Compute loss
        loss = loss_fn(output, mask)
        average_loss += loss.item()
        # Optimize model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # Return average loss
    return average_loss / len(train_loader)

def test_model(model, device, test_loader, jaccard, dice):
    # Initalize average jaccard and dice
    average_jaccard = 0
    average_dice = 0
    # Test the model
    model.eval()
    for batch_idx, (X,y) in enumerate(test_loader):
        # Get batch
        image, mask = X.to(device), y.to(device)
        # Get results
        output = model(image)
        average_jaccard += jaccard(torch.where(output > 0.5, 1, 0),torch.where(mask > 0.50, 1, 0)).item()
        average_dice += dice(torch.where(output > 0.5, 1, 0),torch.where(mask > 0.50, 1, 0)).item()
    # Get average of dice and jaccard scores
    average_jaccard /= len(test_loader)
    average_dice /= len(test_loader)

    # Return values
    return average_jaccard, average_dice