from scripts.train_scripts.train_sr_gnn import train_sr_gnn
from scripts.train_scripts.train_sr_gnn_attn import train_sr_gnn_attn
from scripts.train_scripts.train_sr_gnn_attn_agg import train_sr_gnn_att_agg, train_sr_gnn_att_agg_with_onehot
from scripts.train_scripts.train_simple_sr_gnn_attn import train_simple_sr_gnn_attn
import torch

def train_model(model_name, model_params, output_folder_artifacts, top_k = [20], experiment_hyp_combinat_name = None, resume=None, task="train", best_checkpoint_path=None):
    
    # This function calls all needed methods of the corresponding architecture Class, and can do it differently depending on the architecture.
    train_dataset = None
    eval_dataset = None
    test_dataset = None

    if task == "train":
        print("Loading dataset...")
        train_dataset = torch.load(output_folder_artifacts+"train_dataset.pth", weights_only=False)
        print("Train dataset loaded.")
        eval_dataset = torch.load(output_folder_artifacts+"val_dataset.pth", weights_only=False)
        print("Eval dataset loaded.")
        print(f"Training {model_name}...")
    elif task == "test":
        print("Loading test dataset...")
        test_dataset = torch.load(output_folder_artifacts+"test_dataset.pth", weights_only=False)
        print("Test dataset loaded.")
        print(f"Testing {model_name}...")

    
    # Define preprocessing pipeline for each architecture
    if model_name in {"sr_gnn","sr_gnn_test_mockup","graph_with_embeddings"}:
        train_sr_gnn(train_dataset=train_dataset, eval_dataset=eval_dataset, test_dataset=test_dataset, model_params=model_params, output_folder_artifacts=output_folder_artifacts,top_k=top_k, experiment_hyp_combinat_name=experiment_hyp_combinat_name, resume=resume, task=task, best_checkpoint_path=best_checkpoint_path)
    elif model_name in {"graph_with_embeddings_and_attention"}:
        train_sr_gnn_attn(train_dataset=train_dataset, eval_dataset=eval_dataset, model_params=model_params, output_folder_artifacts=output_folder_artifacts, top_k=top_k, experiment_hyp_combinat_name=experiment_hyp_combinat_name, resume=resume)
    elif model_name in {"graph_with_embeddings_and_attentional_aggregation"}:
        train_sr_gnn_att_agg(train_dataset=train_dataset, eval_dataset=eval_dataset, model_params=model_params, output_folder_artifacts=output_folder_artifacts, top_k=top_k, experiment_hyp_combinat_name=experiment_hyp_combinat_name, resume=resume)
    elif model_name in {"graph_with_encoding_and_attentional_aggregation"}:
        train_sr_gnn_att_agg_with_onehot(train_dataset=train_dataset, eval_dataset=eval_dataset, model_params=model_params, output_folder_artifacts=output_folder_artifacts, top_k=top_k, experiment_hyp_combinat_name=experiment_hyp_combinat_name, resume=resume)
    elif model_name in {"graph_with_attention"}:
        train_simple_sr_gnn_attn(train_dataset=train_dataset, eval_dataset=eval_dataset, model_params=model_params, output_folder_artifacts=output_folder_artifacts, top_k=top_k, experiment_hyp_combinat_name=experiment_hyp_combinat_name, resume=resume)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")