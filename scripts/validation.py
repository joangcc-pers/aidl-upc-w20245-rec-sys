from scripts.evaluate_scripts.evaluate_sr_gnn import evaluate_sr_gnn

def evaluate(model_name, model, split_loader):
    print(f"Evaluating {model_name} in validation split...")

    #TODO: input model object
    # Initialize the model with the same architecture
    model = SR_GNN_attn(hidden_dim=training_params["hidden_dim"],
                    num_iterations=training_params["num_iterations"],
                    num_items=num_values_for_node_embedding["num_items"],
                    embedding_dim=training_params["embedding_dim"],
                    num_categories=num_values_for_node_embedding["num_categories"],
                    num_sub_categories=num_values_for_node_embedding["num_sub_categories"],
                    num_elements=num_values_for_node_embedding["num_elements"],
                    num_brands=num_values_for_node_embedding["num_brands"]
                    )

    # Load the saved weights
    model.load_state_dict(torch.load(output_folder_artifacts))
    

    # Define preprocessing pipeline for each architecture
    if model_name in ["graph_with_embeddings", "graph_with_embeddings_and_attention"]:
        evaluate_sr_gnn(model, split_loader, top_k_values=[5, 10])
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

