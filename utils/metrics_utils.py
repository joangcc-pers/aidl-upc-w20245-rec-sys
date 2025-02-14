import torch

def compute_metrics(predictions, targets, top_k):
    metrics = {}
    for k in top_k:
        # Get top-k predictions
        _, top_indices = torch.topk(predictions, k)
        
        # Recall@k
        hits = torch.zeros_like(targets, dtype=torch.float)
        for i, target in enumerate(targets):
            hits[i] = target in top_indices[i]
        recall = hits.mean().item()
        metrics[f'recall@{k}'] = recall
        
        # Precision@k
        precision = recall / k
        metrics[f'precision@{k}'] = precision
        
        # MRR@k
        mrr = 0
        for i, target in enumerate(targets):
            target_rank = torch.where(top_indices[i] == target)[0]
            if len(target_rank) > 0:
                mrr += 1.0 / (target_rank.item() + 1)
        metrics[f'mrr@{k}'] = mrr / len(targets)
    
    return metrics

def print_metrics(total_epochs, current_epoch, top_k, total_loss, metrics, task = ""):
    metrics_str = [f"epoch={current_epoch + 1}/{total_epochs}", f"Loss={total_loss:.4f}"]
    
    for k in top_k:
        metrics_str.extend([
            f"R@{k}={metrics[f'recall@{k}']:.4f}",
            f"P@{k}={metrics[f'precision@{k}']:.4f}",
            f"MRR@{k}={metrics[f'mrr@{k}']:.4f}"
        ])
    
    print(f"{task}: {' | '.join(metrics_str)}")