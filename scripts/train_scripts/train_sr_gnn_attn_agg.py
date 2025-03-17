from models.sr_gnn_attn_agg import SR_GNN_att_agg
from models.sr_gnn_attn_agg import SR_GNN_att_agg_with_onehot
import torch.optim as optim
import torch.nn as nn
import json
import os
from scripts.collate_fn import collate_fn
from torch.utils.data import DataLoader
from scripts.train_scripts.train_model_utils import print_model_parameters, train_epoch, eval_epoch, test_epoch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
from torch.utils.tensorboard import SummaryWriter

def train_sr_gnn_att_agg(
        model_params,
        train_dataset,
        eval_dataset,                   
        test_dataset = None,
        output_folder_artifacts=None,
        top_k=[20],
        experiment_hyp_combinat_name=None,
        task="train",   
        resume=None,
        best_checkpoint_path=None
):
    if model_params is None:
        raise ValueError("model_params cannot be None")
    if task == "train" and train_dataset is None:
        raise ValueError("Train dataset cannot be None")
    if task == "train" and eval_dataset is None: 
        raise ValueError("Eval dataset cannot be None")
    if task == "test" and test_dataset is None:
        raise ValueError("Test dataset cannot be None if task is 'test'")
    if task == "test" and best_checkpoint_path is None:
        raise ValueError("Best checkpoint path cannot be None if task is 'test'")
    if task == "test" and resume is not None:
        raise ValueError("Resume not available if task is 'test'")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if experiment_hyp_combinat_name is not None:
        output_folder_artifacts_with_exp_hyp_cmb_name = os.path.join(output_folder_artifacts, experiment_hyp_combinat_name)
    else :
        output_folder_artifacts_with_exp_hyp_cmb_name = output_folder_artifacts

    if task == "test":
        output_folder_artifacts_with_exp_hyp_cmb_name = os.path.join(output_folder_artifacts_with_exp_hyp_cmb_name, "test")

    # Crear carpeta de logs para TensorBoard
    log_dir = os.path.join(output_folder_artifacts_with_exp_hyp_cmb_name, "logs")  # Guardar los datos para TensorBoard aquí
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)  # Inicializar TensorBoard (guarda los valores de pérdida y métricas en archivos de log)

    # Read JSON file with training parameters at experiments/sr_gnn_mockup/model_params.json
    # Combine the directory and the file name
    emmbedding_values_file_path = os.path.join(output_folder_artifacts, "num_values_for_node_embedding.json")
   
    with open(emmbedding_values_file_path, "r") as f:
        num_values_for_node_embedding = json.load(f)

    if task == "train":
        train_dataloader = DataLoader(dataset=train_dataset,
                                batch_size=model_params.get("batch_size"),
                                shuffle=model_params.get("shuffle"),
                                collate_fn=collate_fn,
                                pin_memory=(device.type=="cuda")
                                )
        eval_dataloader = DataLoader(dataset=eval_dataset,
                                batch_size=model_params.get("batch_size"),
                                shuffle=False,
                                collate_fn=collate_fn
                                )
    elif task == "test":
        test_dataloader = DataLoader(dataset=test_dataset,
                                batch_size=model_params.get("batch_size"),
                                shuffle=False,
                                collate_fn=collate_fn
                                )

    model = SR_GNN_att_agg(hidden_dim=model_params["hidden_dim"],
                   num_iterations=model_params["num_iterations"],
                   num_items=num_values_for_node_embedding["num_items"],
                   embedding_dim=model_params["embedding_dim"],
                   num_categories=num_values_for_node_embedding["num_categories"],
                   num_sub_categories=num_values_for_node_embedding["num_sub_categories"],
                   num_elements=num_values_for_node_embedding["num_elements"],
                   num_brands=num_values_for_node_embedding["num_brands"],
                   dropout_rate=model_params["dropout_rate"]
                   )
    model = model.to(device)

    print_model_parameters(model)

    if model_params["optimizer"] == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=model_params["lr"], weight_decay=model_params["weight_decay"])
    else:
        raise ValueError(f"Unsupported optimizer: {model_params['optimizer']}")

    criterion = nn.CrossEntropyLoss()

    epochs = model_params["epochs"]

    scheduler = None
    if model_params.get("use_scheduler", True):
        print("Using scheduler")
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)
    else:
        print("Not using scheduler")

    last_checkpoint_epoch = 0
    if resume is not None:
        # Get into output_folder_artifacts_with_exp_hyp_cmb_name, check if there is a trained_model file with the epoch number, and get the maximum epoch number
        for file in os.listdir(output_folder_artifacts_with_exp_hyp_cmb_name):
            # Check if file is a trained_model file and get the max epoch number
            if file.startswith("trained_model_") and file.endswith(".pth"):
                epoch_number = int(file.split("_")[2].split(".")[0])
                if epoch_number > last_checkpoint_epoch:
                    last_checkpoint_epoch = epoch_number
        
        if last_checkpoint_epoch > 0:
            model.load_state_dict(torch.load(output_folder_artifacts_with_exp_hyp_cmb_name + f"/trained_model_{str(last_checkpoint_epoch).zfill(4)}.pth", weights_only=False))
            print(f"Model checkpoint loaded from {output_folder_artifacts_with_exp_hyp_cmb_name + f'/trained_model_{str(last_checkpoint_epoch).zfill(4)}.pth'}")

    if task == "test":
        print("Loading best checkpoint...")
        model.load_state_dict(torch.load(best_checkpoint_path, weights_only=False))
        print(f"Model checkpoint loaded from {best_checkpoint_path}")

        eval_loss, eval_metrics = test_epoch(model, test_dataloader, criterion, top_k=top_k, device=device)
    else:
        # For loop to train the model from first epoch to the last epoch
        for epoch in range(last_checkpoint_epoch, epochs):
            print("----------------------------------")
            if scheduler:
                print(f"Current scheduler-managed lr: {scheduler.get_last_lr()}")
            
            # Entrenamiento y evaluación por época
            train_loss, train_metrics = train_epoch(model, train_dataloader, optimizer, criterion, total_epochs=epochs, current_epoch=epoch, top_k=top_k, device=device)
            eval_loss, eval_metrics = eval_epoch(model, eval_dataloader, criterion, total_epochs=epochs, current_epoch=epoch, top_k=top_k, device=device)

            if scheduler: 
                scheduler.step(eval_loss)

            # Registrar pérdidas y métricas en TensorBoard
            writer.add_scalar("Loss/Train", train_loss, epoch)
            writer.add_scalar("Loss/Validation", eval_loss, epoch)

            for k, v in train_metrics.items():
                writer.add_scalar(f"Train/{k}", v, epoch)

            for k, v in eval_metrics.items():
                writer.add_scalar(f"Validation/{k}", v, epoch)

            # Guardar modelo
            intermediate_model_path = f"trained_model_{str(epoch+1).zfill(4)}.pth"
            torch.save(model.state_dict(), output_folder_artifacts_with_exp_hyp_cmb_name + f"/{intermediate_model_path}")
            print(f"Model for epoch {epoch+1} saved at {intermediate_model_path}")

        # Guardar el modelo final
        torch.save(model.state_dict(), output_folder_artifacts_with_exp_hyp_cmb_name+"/trained_model.pth")
        print(f"Trained model saved at {output_folder_artifacts_with_exp_hyp_cmb_name+'/trained_model.pth'}")

        writer.close()  # Cerrar TensorBoard correctamente

def train_sr_gnn_att_agg_with_onehot(
        model_params,
        train_dataset,
        eval_dataset,
        test_dataset = None,    
        output_folder_artifacts=None,
        top_k=[20],
        experiment_hyp_combinat_name=None,
        task="train",
        resume=None,
        best_checkpoint_path=None
):
    if model_params is None:
        raise ValueError("model_params cannot be None")
    if task == "train" and train_dataset is None:
        raise ValueError("Train dataset cannot be None")
    if task == "train" and eval_dataset is None: 
        raise ValueError("Eval dataset cannot be None")
    if task == "test" and test_dataset is None:
        raise ValueError("Test dataset cannot be None if task is 'test'")
    if task == "test" and best_checkpoint_path is None:
        raise ValueError("Best checkpoint path cannot be None if task is 'test'")
    if task == "test" and resume is not None:
        raise ValueError("Resume not available if task is 'test'")
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if experiment_hyp_combinat_name is not None:
        output_folder_artifacts_with_exp_hyp_cmb_name = os.path.join(output_folder_artifacts, experiment_hyp_combinat_name)
    else :
        output_folder_artifacts_with_exp_hyp_cmb_name = output_folder_artifacts

    if task == "test":
        output_folder_artifacts_with_exp_hyp_cmb_name = os.path.join(output_folder_artifacts_with_exp_hyp_cmb_name, "test")

    # Crear carpeta de logs para TensorBoard
    log_dir = os.path.join(output_folder_artifacts_with_exp_hyp_cmb_name, "logs")  # Guardar los datos para TensorBoard aquí
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)  # Inicializar TensorBoard (guarda los valores de pérdida y métricas en archivos de log)

    # Read JSON file with training parameters at data/processed/sr_gnn_mockup/training_params.json
    # Combine the directory and the file name
    emmbedding_values_file_path = os.path.join(output_folder_artifacts, "num_values_for_node_embedding.json")
   
    with open(emmbedding_values_file_path, "r") as f:
        num_values_for_node_embedding = json.load(f)

    if task == "train":
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
    elif task == "test":
        test_dataloader = DataLoader(dataset=test_dataset,
                                batch_size=model_params.get("batch_size"),
                                shuffle=False,
                                collate_fn=collate_fn
                                )

    # Initialize the model, optimizer and loss function

    model = SR_GNN_att_agg_with_onehot(hidden_dim=model_params["hidden_dim"],
                   num_iterations=model_params["num_iterations"],
                   num_items=num_values_for_node_embedding["num_items"],
                   num_categories=num_values_for_node_embedding["num_categories"],
                   num_sub_categories=num_values_for_node_embedding["num_sub_categories"],
                   num_elements=num_values_for_node_embedding["num_elements"],
                   num_brands=num_values_for_node_embedding["num_brands"],
                   dropout_rate=model_params.get("dropout_rate")
                   )
    model = model.to(device)

    print_model_parameters(model)

    if model_params["optimizer"] == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=model_params["lr"], weight_decay=model_params["weight_decay"])
    else:
        raise ValueError(f"Unsupported optimizer: {model_params['optimizer']}")

    criterion = nn.CrossEntropyLoss()

    epochs = model_params["epochs"]

    scheduler = None
    if model_params.get("use_scheduler", True) and task == "train":
        print("Using scheduler")
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)
    else:
        print("Not using scheduler")

    last_checkpoint_epoch = 0

    if resume is not None:
        # Get into output_folder_artifacts_with_exp_hyp_cmb_name, check if there is a trained_model file with the epoch number, and get the maximum epoch number
        for file in os.listdir(output_folder_artifacts_with_exp_hyp_cmb_name):
            # Check if file is a trained_model file and get the max epoch number
            if file.startswith("trained_model_") and file.endswith(".pth"):
                epoch_number = int(file.split("_")[2].split(".")[0])
                if epoch_number > last_checkpoint_epoch:
                    last_checkpoint_epoch = epoch_number
        
        if last_checkpoint_epoch > 0:
            model.load_state_dict(torch.load(output_folder_artifacts_with_exp_hyp_cmb_name + f"/trained_model_{str(last_checkpoint_epoch).zfill(4)}.pth", weights_only=False))
            print(f"Model checkpoint loaded from {output_folder_artifacts_with_exp_hyp_cmb_name + f'/trained_model_{str(last_checkpoint_epoch).zfill(4)}.pth'}")

    if task == "test":
        print("Loading best checkpoint...")
        model.load_state_dict(torch.load(best_checkpoint_path, weights_only=False))
        print(f"Model checkpoint loaded from {best_checkpoint_path}")

        eval_loss, eval_metrics = test_epoch(model, test_dataloader, criterion, top_k=top_k, device=device)
    else:
        # For loop to train the model from first epoch to the last epoch
        for epoch in range(last_checkpoint_epoch, epochs):
            print("----------------------------------")
            if scheduler:
                print(f"Current scheduler-managed lr: {scheduler.get_last_lr()}")

            train_loss, train_metrics = train_epoch(model, train_dataloader, optimizer, criterion, total_epochs=epochs, current_epoch=epoch, top_k=top_k, device=device)
            eval_loss, eval_metrics = eval_epoch(model, eval_dataloader, criterion, total_epochs=epochs, current_epoch=epoch, top_k=top_k, device=device)

            if scheduler: 
                scheduler.step(eval_loss)

            # Registrar pérdidas y métricas en TensorBoard
            writer.add_scalar("Loss/Train", train_loss, epoch)
            writer.add_scalar("Loss/Validation", eval_loss, epoch)

            for k, v in train_metrics.items():
                writer.add_scalar(f"Train/{k}", v, epoch)

            for k, v in eval_metrics.items():
                writer.add_scalar(f"Validation/{k}", v, epoch)

            # Guardar modelo
            intermediate_model_path = f"trained_model_{str(epoch+1).zfill(4)}.pth"
            torch.save(model.state_dict(), output_folder_artifacts_with_exp_hyp_cmb_name + f"/{intermediate_model_path}")
            print(f"Model for epoch {epoch+1} saved at {intermediate_model_path}")

        # Guardar el modelo final
        torch.save(model.state_dict(), output_folder_artifacts_with_exp_hyp_cmb_name+"/trained_model.pth")
        print(f"Trained model saved at {output_folder_artifacts_with_exp_hyp_cmb_name+'/trained_model.pth'}")

        writer.close()  # Cerrar TensorBoard correctamente


