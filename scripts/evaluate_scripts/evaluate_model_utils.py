import torch
import gc
from tqdm import tqdm
from utils.metrics_utils import compute_precision_and_recall, compute_mrr


def evaluate_model_epoch(model, split_loader, criterion, device=None, top_k_values=[5, 10]):


    """
    Evaluate the model with different hyperparameters on the validation set using different values of K (e.g., 5, 10) and print the results.

    Args:
        model: The trained model.
        split_loader: The DataLoader to evaluate.
        criterion: the loss function
        top_k_values: List of top-K values to evaluate (e.g., [5, 10]).
    """
    model.eval()
    total_loss = 0

    # Initialize accumulators
    total_precision = {k: 0 for k in top_k}
    total_recall = {k: 0 for k in top_k}
    total_mrr = {k: 0 for k in top_k}
    num_batches = len(dataloader)
    

    with torch.no_grad():
        for batch in tqdm(split_loader, "Evaluation Epoch"):
            batch = batch.to(device)
            # Get the predicted scores
            out = model(batch, device)  # [batch_size, num_items]

            # Compute loss
            target = batch.y.to(device)
            loss = criterion(out, target)
            total_loss += loss.item()

            # Collect garbage after processing the batch
            gc.collect()

            predictions = out.detach().cpu()
            targets = target.cpu()

            precision, recall = compute_precision_and_recall(predictions, targets, top_k)
            mrr = compute_mrr(predictions, targets, top_k)

            # Accumulate metrics
            for k in top_k:
                total_precision[k] += precision[k]
                total_recall[k] += recall[k]
                total_mrr[k] += mrr[k]

    # Average the metrics over all batches
    avg_precision = {k: total_precision[k] / num_batches for k in top_k}
    avg_recall = {k: total_recall[k] / num_batches for k in top_k}
    avg_mrr = {k: total_mrr[k] / num_batches for k in top_k}
    avg_loss = total_loss / num_batches

    return avg_loss, avg_precision, avg_recall, avg_mrr

