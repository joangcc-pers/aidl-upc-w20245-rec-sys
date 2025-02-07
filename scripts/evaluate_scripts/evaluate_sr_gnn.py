import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score
import numpy as np

def evaluate_sr_gnn(model, test_loader, top_k_values=[5, 10]):
    """
    Evaluate the model with different hyperparameters on the validation set using different values of K (e.g., 5, 10) and print the results.

    Args:
        model: The trained model.
        test_loader: The DataLoader for the test dataset.
        top_k_values: List of top-K values to evaluate (e.g., [5, 10]).
    """
    for top_k in top_k_values:
        print(f"\nEvaluating with Top-{top_k} Recommendations...")
        mrr_at_k, recall_at_k, precision_at_k = evaluate_model(model, test_loader, top_k=top_k)

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
    mrr_at_k_total = 0
    recall_at_k_total = 0
    precision_at_k_total = 0
    total_samples = 0

    with torch.no_grad():
        for batch in test_loader:
            # Get the predicted scores
            out = model(batch)  # [batch_size, num_items]

            # Get the ground truth (target)
            y_true = batch.y.cpu().numpy()  # The true label (target item)
            _, top_k_preds = torch.topk(out, top_k, dim=1, largest=True, sorted=True)

            # Get the target items index
            target_index = np.where(top_k_preds.cpu().numpy() == y_true[:, None])[1]

            # Calculate MRR@K 
            mrr_at_k = np.mean(1 / (target_index + 1))  # +1 for 1-based indexing
            mrr_at_k_total += mrr_at_k

            # Calculate Recall@K 
            recall_at_k = np.mean(np.isin(target_index, range(top_k)))  # 1 if target item is in top K
            recall_at_k_total += recall_at_k

            # Calculate Precision@K 
            precision_at_k = np.mean(np.isin(target_index, range(top_k)))  # Precision is same for Recall in this case
            precision_at_k_total += precision_at_k

            total_samples += 1

    # Averaging over all batches
    mrr_at_k_avg = mrr_at_k_total / total_samples
    recall_at_k_avg = recall_at_k_total / total_samples
    precision_at_k_avg = precision_at_k_total / total_samples

    print(f"Test Results at Top-{top_k}:")
    print(f"Mean Reciprocal Rank (MRR@{top_k}): {mrr_at_k_avg:.4f}")
    print(f"Recall@{top_k}: {recall_at_k_avg:.4f}")
    print(f"Precision@{top_k}: {precision_at_k_avg:.4f}")

    return mrr_at_k_avg, recall_at_k_avg, precision_at_k_avg
