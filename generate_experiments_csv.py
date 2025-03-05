import os
import glob
import re
import tensorflow as tf
import csv

def extract_metrics_from_event_file(event_file):
    metrics = {}
    for event in tf.compat.v1.train.summary_iterator(event_file):
        for value in event.summary.value:
            if value.tag not in metrics:
                metrics[value.tag] = []
            metrics[value.tag].append((event.step, value.simple_value))
    return metrics

def extract_hyperparameters_from_folder_name(folder_name):
    pattern = r"wd([0-9e\.-]+)_dr([0-9\.]+)_lr([0-9e\.-]+)"
    match = re.search(pattern, folder_name)
    if match:
        weight_decay = match.group(1)
        dropout_rate = match.group(2)
        learning_rate = match.group(3)
        return weight_decay, dropout_rate, learning_rate
    return None, None, None

def generate_report(events_dir, output_file):
    fieldnames = ['model', 'weight_decay', 'dropout_rate', 'learning_rate', 'epoch', 'metric', 'value']
    rows = []
    
    event_files = glob.glob(os.path.join(events_dir, "**", "events.out.tfevents.*"), recursive=True)
    
    # Agrupar archivos por carpeta
    logs_folders = {}
    for event_file in event_files:
        logs_folder = os.path.dirname(event_file)
        if logs_folder not in logs_folders:
            logs_folders[logs_folder] = []
        logs_folders[logs_folder].append(event_file)
    
    # Procesar el archivo más reciente en cada carpeta
    for logs_folder, files in logs_folders.items():
        latest_file = max(files, key=os.path.getmtime)
        parent_folder = os.path.basename(os.path.dirname(logs_folder))
        weight_decay, dropout_rate, learning_rate = extract_hyperparameters_from_folder_name(parent_folder)
        
        metrics = extract_metrics_from_event_file(latest_file)
        for tag, values in metrics.items():
            for step, value in values:
                rows.append({
                    'model': parent_folder,
                    'weight_decay': weight_decay,
                    'dropout_rate': dropout_rate,
                    'learning_rate': learning_rate,
                    'epoch': step,
                    'metric': tag,
                    'value': value
                })
    
    # Guardar los resultados en un archivo CSV
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

# Ejemplo de uso
generate_report("experiments/graph_with_embeddings_and_attentional_aggregation", "experiments_report.csv")