import torch

def train_model_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    all_predictions = []
    all_targets = []

    for batch in dataloader:
        optimizer.zero_grad()
        out = model(batch)  

        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Store predictions and targets for metric computation
        predictions = out.detach()
        all_predictions.append(predictions)
        all_targets.append(batch.y)
    return torch.cat(all_predictions), torch.cat(all_targets), total_loss