import torch
from tqdm import tqdm

def train_model_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_predictions = []
    all_targets = []

    for batch in tqdm(dataloader, desc="Training Epoch"):
        batch = batch.to(device)

        optimizer.zero_grad()
        out = model(batch, device)

        target = batch.y.to(device)
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Store predictions and targets for metric computation
        predictions = out.detach()
        all_predictions.append(predictions)
        all_targets.append(batch.y)

    return torch.cat(all_predictions), torch.cat(all_targets), total_loss