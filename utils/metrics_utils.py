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


def compute_precision(predictions, targets, top_k=[10, 20]):
    with torch.no_grad():
        batch_size = targets.size(0)
        max_k = max(top_k)
        _, pred = predictions.topk(max_k, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))

        precision = {}
        for k in top_k:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            precision[k] = correct_k.mul_(100.0 / batch_size)
        return precision
    
def compute_recall(predictions, targets, top_k=[10, 20]):
    with torch.no_grad():
        batch_size = targets.size(0)
        max_k = max(top_k)
        _, pred = predictions.topk(max_k, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))

        recall = {}
        for k in top_k:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            total_k = targets.size(0) 
            recall[k] = correct_k.mul_(100.0 / total_k)
        return recall
    
def compute_mrr(predictions, targets, top_k=[10, 20]):
    with torch.no_grad():
        batch_size = targets.size(0)
        mrr = {}
        
        for k in top_k:
            _, pred = predictions.topk(k, dim=1, largest=True, sorted=True)
            pred = pred.t()
            ranks = []
            for i in range(batch_size):
                true_index = (pred[:, i] == targets[i]).nonzero(as_tuple=True)[0]
                if true_index.numel() > 0:
                    rank = true_index.item() + 1
                    ranks.append(1.0 / rank)
                else:
                    ranks.append(0.0)  
            
            mrr[k] = sum(ranks) / batch_size
        return mrr