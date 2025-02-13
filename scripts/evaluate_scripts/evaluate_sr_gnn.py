import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score
import numpy as np
from scripts.collate_fn import collate_fn
from torch.utils.data import DataLoader
import os


def evaluate_sr_gnn(model, split_loader, criterion, top_k_values=[5, 10]):
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
        for batch in split_loader:
            # Get the predicted scores
            out = model(batch)  # [batch_size, num_items]

            # Compute loss
            loss = criterion(out, batch.y)
            total_loss += loss.item()

            # Get top predictions
            _, top_k_preds = torch.topk(out, max(top_k_values), dim=1, largest=True, sorted=True)
            
            # Store predictions and targets (keeping them as tensors)
            all_predictions.append(top_k_preds)
            all_targets.append(batch.y)

    return torch.cat(all_predictions), torch.cat(all_targets), total_loss

