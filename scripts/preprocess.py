import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, normalize
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from scripts.preprocess_graph_with_embeddings import preprocess_graph_with_embeddings


def preprocess_data(model_name, input_folder_path, output_path, preprocessing_params):
    
    print(f"Preprocessing data for {model_name}...")

    # Define preprocessing pipeline for each architecture
    if model_name == "graph_with_embeddings":
        data = preprocess_graph_with_embeddings(input_folder_path, preprocessing_params)
    # elif model_name == "hierarchical_rnn":
    #     data = preprocess_for_hierarchical_rnn(data, preprocessing_params)
    # elif model_name == "temporal_transformer":
    #     data = preprocess_for_temporal_transformer(data, preprocessing_params)
    # elif model_name == "kmeans_base_model":
    #     data = preprocess_for_kmeans(data, preprocessing_params)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    # Save cleaned and preprocessed data
    data.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")


# def preprocess_for_hierarchical_rnn(data, preprocessing_params):
#     """Preprocessing pipeline for Hierarchical RNN."""
#     print("Applying preprocessing for Hierarchical RNN...")
    
#     # Scaling
#     if preprocessing_params.get("scaling") == "minmax":
#         scaler = MinMaxScaler()
#         data[data.select_dtypes(include=["float64", "int64"]).columns] = scaler.fit_transform(
#             data.select_dtypes(include=["float64", "int64"])
#         )
#     elif preprocessing_params.get("scaling") == "standard":
#         scaler = StandardScaler()
#         data[data.select_dtypes(include=["float64", "int64"]).columns] = scaler.fit_transform(
#             data.select_dtypes(include=["float64", "int64"])
#         )

#     # Normalization
#     if preprocessing_params.get("normalization"):
#         data[data.select_dtypes(include=["float64", "int64"]).columns] = normalize(
#             data.select_dtypes(include=["float64", "int64"])
#         )

#     return data


# def preprocess_for_temporal_transformer(data, preprocessing_params):
#     """Preprocessing pipeline for Temporal Transformer."""
#     print("Applying preprocessing for Temporal Transformer...")
    
#     # Transformers typically benefit from scaled continuous data
#     if preprocessing_params.get("scaling") == "standard":
#         scaler = StandardScaler()
#         data[data.select_dtypes(include=["float64", "int64"]).columns] = scaler.fit_transform(
#             data.select_dtypes(include=["float64", "int64"])
#         )

#     # Transformers might not need normalization but could use sequence-specific transformations
#     if preprocessing_params.get("normalization"):
#         data[data.select_dtypes(include=["float64", "int64"]).columns] = normalize(
#             data.select_dtypes(include=["float64", "int64"])
#         )

#     return data


# def preprocess_for_kmeans(data, preprocessing_params):
#     """Preprocessing pipeline for KMeans clustering."""
#     print("Applying preprocessing for KMeans...")

#     # Clustering models like KMeans often benefit from MinMax scaling
#     if preprocessing_params.get("scaling") == "minmax":
#         scaler = MinMaxScaler()
#         data[data.select_dtypes(include=["float64", "int64"]).columns] = scaler.fit_transform(
#             data.select_dtypes(include=["float64", "int64"])
#         )

#     return data
