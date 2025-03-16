import gc
from utils.metrics_utils import compute_precision_and_recall, compute_mrr, aggregate_metrics, print_metrics
from scripts.evaluate_scripts.evaluate_model_utils import evaluate_model_epoch
from tqdm import tqdm

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
 
def print_model_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())  # Total parameters
    learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)  # Learnable parameters

    print(f"Total parameters: {total_params}")
    print(f"Learnable parameters: {learnable_params}")


def train_epoch(model, dataloader, optimizer, criterion, total_epochs, current_epoch, top_k=[20], device=None):
    avg_loss, avg_precision, avg_recall, avg_mrr = train_model_epoch(model, dataloader, optimizer, criterion, device, top_k=top_k)

    metrics = aggregate_metrics(avg_loss, avg_precision, avg_recall, avg_mrr)
        
    print_metrics(total_epochs, current_epoch, top_k, avg_loss, metrics, task="Training")
    return avg_loss, metrics  # Retornar pérdida y métricas

def eval_epoch(model, eval_dataloader, criterion, total_epochs, current_epoch, top_k=[20], device=None):
    avg_loss, avg_precision, avg_recall, avg_mrr = evaluate_model_epoch(model, eval_dataloader, criterion, device, top_k)

    metrics = aggregate_metrics(avg_loss, avg_precision, avg_recall, avg_mrr)
        
    print_metrics(total_epochs, current_epoch, top_k, avg_loss, metrics, task="Evaluate")
    return avg_loss, metrics  # Retornar pérdida y métricas

def test_epoch(model, eval_dataloader, criterion, top_k=[20], device=None):
    avg_loss, avg_precision, avg_recall, avg_mrr = evaluate_model_epoch(model, eval_dataloader, criterion, device, top_k)

    metrics = aggregate_metrics(avg_loss, avg_precision, avg_recall, avg_mrr)
    
    print_metrics(1, 0, top_k, avg_loss, metrics, task="Test")
    return avg_loss, metrics  # Retornar pérdida y métricas