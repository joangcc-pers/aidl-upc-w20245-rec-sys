import torch
import gc

from utils.metrics_utils import compute_precision, compute_recall, compute_mrr


def train_model_epoch(model, dataloader, optimizer, criterion, device, top_k=[10,20]):
    model.train()
    total_loss = 0

    # Initialize accumulators
    total_precision = {k: 0 for k in top_k}
    total_recall = {k: 0 for k in top_k}
    total_mrr = {k: 0 for k in top_k}
    num_batches = len(dataloader)

    for batch in tqdm(dataloader, desc="Training Epoch"):
        batch = batch.to(device)

        optimizer.zero_grad()
        out = model(batch, device)

        target = batch.y.to(device)
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Collect garbage after processing the batch
        gc.collect()

        predictions = out.detach().cpu()
        targets = batch.y.cpu()

        precision = compute_precision(predictions, targets, top_k)
        recall = compute_recall(predictions, targets, top_k)
        mrr = compute_mrr(predictions, targets, top_k)

        # Accumulate metrics
        for k in top_k:
            total_precision[k] += precision[k].item()
            total_recall[k] += recall[k].item()
            total_mrr[k] += mrr[k]

     # Average the metrics over all batches
    avg_precision = {k: total_precision[k] / num_batches for k in top_k}
    avg_recall = {k: total_recall[k] / num_batches for k in top_k}
    avg_mrr = {k: total_mrr[k] / num_batches for k in top_k}

    return total_loss, avg_precision, avg_recall, avg_mrr
 
