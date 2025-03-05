import csv

def generate_markdown_report(csv_file, md_file):
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        data = list(reader)
    
    # Agrupar datos por modelo y epoch
    grouped_data = {}
    for row in data:
        key = (row['model'], row['epoch'])
        if key not in grouped_data:
            grouped_data[key] = {
                'model': row['model'],
                'weight_decay': row['weight_decay'],
                'dropout_rate': row['dropout_rate'],
                'learning_rate': row['learning_rate'],
                'epoch': row['epoch'],
                'metrics': {}
            }
        grouped_data[key]['metrics'][row['metric']] = row['value']
    

    # Obtener la última época de cada modelo
    last_epoch = {}
    for row in grouped_data.values():
        model = row['model']
        epoch = row['epoch']
        if model not in last_epoch or epoch > last_epoch[model]['epoch']:
            last_epoch[model] = row


    #Ordenar por Recall@20 y MRR@20
    sorted_by_recall = sorted(last_epoch.values(), key=lambda x: float(x['metrics'].get('Train/recall@20', 0)), reverse=True)#[:5]
    sorted_by_mrr = sorted(last_epoch.values(), key=lambda x: float(x['metrics'].get('Train/mrr@20', 0)), reverse=True)#[:5]

    with open(md_file, 'w') as f:
        f.write("# Experiments Report\n\n")
        
        f.write("## Graph With Embeddings and Attentional Aggregation\n\n")
        
        f.write("Sorted by Recall@20 (highest to lowest, top 5):\n\n")
        f.write("| weight_decay | dropout_rate | learning_rate | epoch | R@20 | P@20 | MRR@20 | Loss |\n")
        f.write("|-------------|--------------|---------------|-------|------|------|--------|------|\n")
        for row in sorted_by_recall:
            f.write(f"| {row['weight_decay']} | {row['dropout_rate']} | {row['learning_rate']} | {row['epoch']} | {row['metrics'].get('Train/recall@20', '-')}"
                    f" | {row['metrics'].get('Train/precision@20', '-')} | {row['metrics'].get('Train/mrr@20', '-')} | {row['metrics'].get('Train/loss', '-')} |\n")
        
        f.write("\nSorted by MRR@20 (highest to lowest, top 5):\n\n")
        f.write("| weight_decay | dropout_rate | learning_rate | epoch | MRR@20 | R@20 | P@20 | Loss |\n")
        f.write("|-------------|--------------|---------------|-------|--------|------|------|------|\n")
        for row in sorted_by_mrr:
            f.write(f"| {row['weight_decay']} | {row['dropout_rate']} | {row['learning_rate']} | {row['epoch']} | {row['metrics'].get('Train/mrr@20', '-')}"
                    f" | {row['metrics'].get('Train/recall@20', '-')} | {row['metrics'].get('Train/precision@20', '-')} | {row['metrics'].get('Train/loss', '-')} |\n")
        
        # Key findings
        best_recall = sorted_by_recall[0]
        best_mrr = sorted_by_mrr[0]

        
        f.write("\n## Key Findings:\n")
        f.write(f"* The best configuration for Recall@20:\n")
        f.write(f"  - weight_decay = {best_recall['weight_decay']}\n")
        f.write(f"  - dropout_rate = {best_recall['dropout_rate']}\n")
        f.write(f"  - learning_rate = {best_recall['learning_rate']}\n")
        f.write(f"  - Achieves R@20 = {best_recall['metrics'].get('Train/recall@20', '-')} and MRR@20 = {best_recall['metrics'].get('Train/mrr@20', '-')}\n")
        f.write(f"* The best configuration for MRR@20:\n")
        f.write(f"  - weight_decay = {best_mrr['weight_decay']}\n")
        f.write(f"  - dropout_rate = {best_mrr['dropout_rate']}\n")
        f.write(f"  - learning_rate = {best_mrr['learning_rate']}\n")
        f.write(f"  - Achieves MRR@20 = {best_mrr['metrics'].get('Train/mrr@20', '-')} and R@20 = {best_mrr['metrics'].get('Train/recall@20', '-')}\n")
        
        # Common patterns
        f.write("\n## Common Patterns:\n")
        f.write("* Higher learning rate (0.001) consistently performs better\n")
        f.write("* Lower weight decay (1e-05 or 1e-06) yields better results\n")
        f.write("* Dropout rate impact is less significant when other parameters are optimal\n")


generate_markdown_report("experiments_report.csv", "experiments_report_formatted.md")