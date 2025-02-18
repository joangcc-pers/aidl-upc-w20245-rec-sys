import torch
import time

def train_model_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_predictions = []
    all_targets = []

    start_time = time.time()

    for batch in dataloader:
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
    
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time for train_model_epoch: {execution_time:.4f} seconds")

    return torch.cat(all_predictions), torch.cat(all_targets), total_loss