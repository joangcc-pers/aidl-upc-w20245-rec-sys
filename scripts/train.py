from scripts.train_scripts.train_sr_gnn import train_sr_gnn
from scripts.train_scripts.train_sr_gnn_attn import train_sr_gnn_attn
from scripts.train_scripts.train_sr_gnn_attn_agg import train_sr_gnn_att_agg
import torch

def train_model(model_name, model_params, output_folder_artifacts, top_k = [20]):
    
    # This function calls all needed methods of the corresponding architecture Class, and can do it differently depending on the architecture.
    print("Loading dataset...")
    train_dataset = torch.load(output_folder_artifacts+"train_dataset.pth", weights_only=False)
    print("Train dataset loaded.")
    eval_dataset = torch.load(output_folder_artifacts+"val_dataset.pth", weights_only=False)

    print(f"Training {model_name}...")
    # Define preprocessing pipeline for each architecture
    if model_name in {"sr_gnn","sr_gnn_test_mockup","graph_with_embeddings"}:
        train_sr_gnn(train_dataset=train_dataset, eval_dataset=eval_dataset, model_params=model_params, output_folder_artifacts=output_folder_artifacts,top_k=top_k)
    elif model_name in {"graph_with_embeddings_and_attention"}:
        train_sr_gnn_attn(train_dataset=train_dataset, eval_dataset=eval_dataset, model_params=model_params, output_folder_artifacts=output_folder_artifacts, top_k=top_k)
    elif model_name in {"graph_with_embeddings_and_attentional_aggregation"}:
        train_sr_gnn_att_agg(train_dataset=train_dataset, eval_dataset=eval_dataset, model_params=model_params, output_folder_artifacts=output_folder_artifacts, top_k=top_k)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")