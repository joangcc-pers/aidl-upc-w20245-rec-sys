from scripts.preprocess_graph_with_embeddings import preprocess_graph_with_embeddings


def preprocess_data(model_name, input_folder_path, output_path, preprocessing_params):
    
    print(f"Preprocessing data for {model_name}...")

    # Define preprocessing pipeline for each architecture
    if model_name == "graph_with_embeddings":
        data = preprocess_graph_with_embeddings(input_folder_path, preprocessing_params)
    #### PLACEHOLDERS: NOT DEVELOPED YET. WE MAY NOT DEVELOP IT ####
    # elif model_name == "hierarchical_rnn":
    #     data = preprocess_for_hierarchical_rnn(data, preprocessing_params)
    # elif model_name == "temporal_transformer":
    #     data = preprocess_for_temporal_transformer(data, preprocessing_params)
    # elif model_name == "kmeans_base_model":
    #     data = preprocess_for_kmeans(data, preprocessing_params)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    # Save cleaned and preprocessed data
    # data.to_csv(output_path, index=False)
    # print(f"Data saved to {output_path}")