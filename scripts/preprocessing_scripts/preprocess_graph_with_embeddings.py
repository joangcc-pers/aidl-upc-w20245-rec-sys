from scripts.preprocessing_scripts.session_graph_embeddings_dataset import SessionGraphEmbeddingsDataset
from scripts.collate_fn import collate_fn
from torch.utils.data import Subset, random_split

import torch

def preprocess_graph_with_embeddings(input_folder_path, output_folder_artifacts, preprocessing_params):
    """Preprocessing pipeline for Graph NN."""
    print("Applying preprocessing for Graph NN with embeddings for categorical features...")

    # Initialize dataset and dataloader
    dataset = SessionGraphEmbeddingsDataset(folder_path=input_folder_path,
                                            output_folder_artifacts=output_folder_artifacts,
                                            start_month=preprocessing_params.get("start_month"),
                                            end_month=preprocessing_params.get("end_month"),
                                            transform=None,
                                            test_sessions_first_n=preprocessing_params.get("test_sessions_first_n"),
                                            limit_to_view_event=preprocessing_params.get("limit_to_view_event"),
                                            drop_listwise_nulls=preprocessing_params.get("drop_listwise_nulls"),
                                            min_products_per_session=preprocessing_params.get("min_products_per_session"),
                                            )
    
    #DEBUGGING:
    # Check shape of one sample
    # Get first sample
    sample = dataset

    print(sample)  # Prints a summary of the graph data object

    # Check individual components
    if hasattr(sample, "product_id_remapped"):
        print(f"product_id_remapped shape: {sample.product_id_remapped.shape}")
    if hasattr(sample, "category"):
        print(f"category shape: {sample.product_id_remapped.shape}")
    if hasattr(sample, "sub_category"):
        print(f"category shape: {sample.sub_category.shape}")
    if hasattr(sample, "element"):
        print(f"category shape: {sample.element.shape}")
    if hasattr(sample, "brand"):
        print(f"category shape: {sample.brand.shape}")
    if hasattr(sample, "price_tensor"):
        print(f"Price tensor shape: {sample.price_tensor.shape}")
    if hasattr(sample, "edge_index"):
        print(f"Edge index shape: {sample.edge_index.shape}")
    if hasattr(sample, "y"):
        print(f"Target shape: {sample.y.shape}")


    train_split = preprocessing_params["train_split"]
    val_split = preprocessing_params["val_split"]
    test_split = preprocessing_params["test_split"]
    split_method=preprocessing_params["split_method"]

    total_size = len(dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = int(test_split * total_size)
    
    if split_method == "random":
        # Perform random splitting
        train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    elif split_method == "temporal":
        if dataset.timestamps is None:
            raise ValueError("Temporal splitting requires timestamps in the dataset.")

        # Sort dataset based on timestamps
        sorted_indices = sorted(range(len(dataset.timestamps)), key=lambda i: dataset.timestamps[i])
        sorted_data = [dataset.data[i] for i in sorted_indices]

        # Perform sequential splitting
        train_dataset = Subset(sorted_data[:train_size])
        val_dataset = Subset(sorted_data[train_size:train_size + val_size])
        test_dataset = Subset(sorted_data[train_size + val_size:])
    
    else:
        raise ValueError(f"Unsupported split method: {split_method}")
    
    torch.save(train_dataset, output_folder_artifacts+f"train_dataset.pth")
    torch.save(val_dataset, output_folder_artifacts+f"val_dataset.pth")
    torch.save(test_dataset, output_folder_artifacts+f"test_dataset.pth")

    print(f"Datasets saved in {output_folder_artifacts}")