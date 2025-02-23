from scripts.preprocessing_scripts.preprocess_graph_with_embeddings import preprocess_graph_with_embeddings
from scripts.preprocessing_scripts.preprocess_graph_with_encoding import preprocess_graph_with_onehot
# from scripts.preprocessing_scripts.preprocess_sr_gnn import preprocess_sr_gnn

def preprocess_data(model_name, input_folder_path, output_folder_artifacts, preprocessing_params):
    
    print(f"Preprocessing data for {model_name}...")

    # Define preprocessing pipeline for each architecture
    if model_name in ["graph_with_embeddings", "graph_with_embeddings_and_attention", "graph_with_embeddings_and_attentional_aggregation"]:
        preprocess_graph_with_embeddings(input_folder_path, output_folder_artifacts, preprocessing_params)
    elif model_name in ["graph_with_encoding", "graph_with_encoding_and_attentional_aggregation"]:
        preprocess_graph_with_onehot(input_folder_path, output_folder_artifacts, preprocessing_params)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")