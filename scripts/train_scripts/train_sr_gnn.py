from models.sr_gnn import SR_GNN
import torch.optim as optim
import torch.nn as nn
import json
import os
from scripts.collate_fn import collate_fn
from torch.utils.data import DataLoader
from scripts.evaluate_scripts.evaluate_sr_gnn import evaluate_sr_gnn
from utils.metrics_utils import compute_metrics, print_metrics
import torch

def train_sr_gnn(
        model_params,
        train_dataset,
        eval_dataset,
        output_folder_artifacts=None,
        top_k=[20]
):
    if model_params is None:
        raise ValueError("model_params cannot be None")
    if train_dataset is None:
        raise ValueError("Train dataset cannot be None")
    if eval_dataset is None: 
        raise ValueError("Eval dataset cannot be None")

    # Read JSON file with training parameters at experiments/sr_gnn_mockup/model_params.json
    # Combine the directory and the file name
    file_path = os.path.join(output_folder_artifacts, "num_values_for_node_embedding.json")
    train_dataloader = DataLoader(dataset=train_dataset,
                            batch_size=model_params.get("batch_size"),
                            shuffle=model_params.get("shuffle"),
                            collate_fn=collate_fn
                            )
    
    eval_dataloader = DataLoader(dataset=eval_dataset,
                            batch_size=model_params.get("batch_size"),
                            shuffle=False,
                            collate_fn=collate_fn
                            )

    # Open and load the JSON file
    with open(file_path, "r") as f:
        num_values_for_node_embedding = json.load(f)

    # Initialize the model, optimizer and loss function

    model = SR_GNN(hidden_dim=model_params["hidden_dim"],
                   num_iterations=model_params["num_iterations"],
                   num_items=num_values_for_node_embedding["num_items"],
                   embedding_dim=model_params["embedding_dim"],
                   num_categories=num_values_for_node_embedding["num_categories"],
                   num_sub_categories=num_values_for_node_embedding["num_sub_categories"],
                   num_elements=num_values_for_node_embedding["num_elements"],
                   num_brands=num_values_for_node_embedding["num_brands"]
                   )

    if model_params["optimizer"] == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=model_params["lr"])
    else:
        raise ValueError(f"Unsupported optimizer: {model_params['optimizer']}")

    criterion = nn.CrossEntropyLoss()

    epochs = model_params["epochs"]

    for epoch in range(epochs):
        print("----------------------------------")
        
        train_epoch(model, train_dataloader, optimizer, criterion, total_epochs=epochs, current_epoch=epoch, top_k=top_k)
        eval_epoch(model, eval_dataloader, criterion, total_epochs=epochs, current_epoch=epoch, top_k= top_k)
        
        # Save the model state_dict for the epoch
        intermediate_model_path = f"trained_model_{str(epoch+1).zfill(4)}.pth"
        torch.save(model.state_dict(), output_folder_artifacts + f"trained_model_{str(epoch+1).zfill(4)}.pth")
        print(f"Model for epoch {epoch+1} saved at {intermediate_model_path}")
    
    #Save the final model implementation
    torch.save(model.state_dict(), output_folder_artifacts+"trained_model.pth")
    print(f"Trained model saved at {output_folder_artifacts+'trained_model.pth'}")

def train_epoch(model, dataloader, optimizer, criterion, total_epochs, current_epoch, top_k=[20]):
    model.train()
    total_loss = 0
    all_predictions = []
    all_targets = []

    for batch in dataloader:
        optimizer.zero_grad()
        out = model(batch)  

        loss = criterion(out, batch.y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Store predictions and targets for metric computation
        predictions = out.detach()
        all_predictions.append(predictions)
        all_targets.append(batch.y)
    
    metrics = compute_metrics(torch.cat(all_predictions), torch.cat(all_targets), top_k)
    
    print_metrics(total_epochs, current_epoch, top_k, total_loss, metrics, task="Training")

def eval_epoch(model, eval_dataloader, criterion, total_epochs, current_epoch, top_k=[20]):
    all_predictions, all_targets, total_loss = evaluate_sr_gnn(model, eval_dataloader, criterion, top_k)

    metrics = compute_metrics(all_predictions, all_targets, top_k)
    
    print_metrics(total_epochs, current_epoch, top_k, total_loss, metrics, task="Validate")
