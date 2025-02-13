import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score
import numpy as np
from scripts.collate_fn import collate_fn
from torch.utils.data import DataLoader
import os


def evaluate_sr_gnn(model, split_loader, top_k_values=[5, 10]):
    """
    Evaluate the model with different hyperparameters on the validation set using different values of K (e.g., 5, 10) and print the results.

    Args:
        model: The trained model.
        split_loader: The DataLoader to evaluate.
        top_k_values: List of top-K values to evaluate (e.g., [5, 10]).
    """
    if model is None:
        raise ValueError("model cannot be None")
    if split_loader is None:
        raise ValueError("split_loader cannot be None")

    for top_k in top_k_values:
        print(f"\nEvaluating with Top-{top_k} Recommendations...")
        mrr_at_k, recall_at_k, precision_at_k = evaluate_model(model, split_loader, top_k=top_k)

        # Print results for this top-k evaluation
        print(f"\n--- Results for Top-{top_k} ---")
        print(f"MRR@{top_k}: {mrr_at_k:.4f}")
        print(f"Recall@{top_k}: {recall_at_k:.4f}")
        print(f"Precision@{top_k}: {precision_at_k:.4f}")

def evaluate_model(model, test_loader, top_k=10):
    """
    Test the model on the test set using the specified metrics.

    Args:
        model: The trained model.
        test_loader: The DataLoader for the test dataset.
        top_k: The top-k recommendations to evaluate.

    Returns:
        mrr_at_k, recall_at_k, precision_at_k
    """
    model.eval()
    samples_reciprocal_ranks = []
    samples_recalls = []
    samples_precisions = []

    with torch.no_grad():
        for batch in test_loader:
            # Get the predicted scores
            out = model(batch)  # [batch_size, num_items]

            # Get the ground truth (target) and reshape it to match the shape of predictions
            y_true = batch.y.cpu().numpy().reshape(-1, 1)  # The true label (target item)

            _, top_k_preds = torch.topk(out, top_k, dim=1, largest=True, sorted=True)
            top_k_preds = top_k_preds.cpu().numpy()
            
            # Per-sample metric computation, as MRR requires computing the reciprocal rank per sample
            for i in range(len(y_true)):
                target = y_true[i][0]
                predictions = top_k_preds[i]
                
                # Get the indexes where target appears in the model predictions
                target_rank = np.where(predictions == target)[0]
                if len(target_rank) > 0:
                    reciprocal_rank = 1.0 / (target_rank[0] + 1)
                    samples_reciprocal_ranks.append(reciprocal_rank)
                else: # No prediction matching target
                    samples_reciprocal_ranks.append(0.0)
                
                hit = int(target in predictions)
                samples_recalls.append(hit)  # Target is in the top-k
                samples_precisions.append(hit / top_k)

    # Calculate final metrics
    mrr_at_k_avg = np.mean(samples_reciprocal_ranks)
    recall_at_k_avg = np.mean(samples_recalls)
    precision_at_k_avg = np.mean(samples_precisions)

    print(f"Test Results at Top-{top_k}:")
    print(f"Mean Reciprocal Rank (MRR@{top_k}): {mrr_at_k_avg:.4f}")
    print(f"Recall@{top_k}: {recall_at_k_avg:.4f}")
    print(f"Precision@{top_k}: {precision_at_k_avg:.4f}")

    return mrr_at_k_avg, recall_at_k_avg, precision_at_k_avg
