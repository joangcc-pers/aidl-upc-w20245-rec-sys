import torch
from tqdm import tqdm

def evaluate_model_epoch(model, dataloader, criterion, device, top_k_values=[5, 10]):
    """
    Evaluate the model with different hyperparameters on the validation set using different values of K (e.g., 5, 10) and print the results.

    Args:
        model: The trained model.
        split_loader: The DataLoader to evaluate.
        criterion: the loss function
        top_k_values: List of top-K values to evaluate (e.g., [5, 10]).
    """
    model.eval()
    all_predictions = []
    all_targets = []
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, "Evaluation Epoch"):
            batch = batch.to(device)
            # Get the predicted scores
            out = model(batch, device)  # [batch_size, num_items]

            # Compute loss
            target = batch.y.to(device)
            loss = criterion(out, target)
            total_loss += loss.item()

            # Get top predictions
            _, top_k_preds = torch.topk(out, max(top_k_values), dim=1, largest=True, sorted=True)
            
            # Store predictions and targets (keeping them as tensors)
            all_predictions.append(top_k_preds)
            all_targets.append(batch.y)

    return torch.cat(all_predictions), torch.cat(all_targets), total_loss

