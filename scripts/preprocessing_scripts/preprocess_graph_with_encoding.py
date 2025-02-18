from torch.utils.data import DataLoader
from scripts.session_graph_encoding_dataset import SessionGraphOneHotDataset
from scripts.collate_fn import collate_fn
from torch.utils.data import Subset, random_split

import torch


def preprocess_graph_with_onehot(input_folder_path, output_folder_artifacts, preprocessing_params):
    """Preprocessing pipeline for Graph NN."""
    print("Applying preprocessing for Graph NN with embeddings for categorical features...")

    # Initialize dataset and dataloader
    dataset = SessionGraphOneHotDataset(folder_path=input_folder_path,
                                            output_folder_artifacts=output_folder_artifacts,
                                            start_month=preprocessing_params.get("start_month"),
                                            end_month=preprocessing_params.get("end_month"),
                                            transform=None,
                                            test_sessions_first_n=preprocessing_params.get("test_sessions_first_n"), 
                                            # embedding_dim=64,
                                            limit_to_view_event=preprocessing_params.get("limit_to_view_event"),
                                            drop_listwise_nulls=preprocessing_params.get("drop_listwise_nulls"),
                                            min_products_per_session=preprocessing_params.get("min_products_per_session")
                                            )
    sample= dataset[0]
    print(sample)
    
    
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
    split_method = preprocessing_params["split_method"]

    total_size = len(dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = int(test_split * total_size)

    if split_method == "random":
        # Perform random splitting
        train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    elif split_method == "temporal":
        # Requires dataset to have been already presorted. Do not use shuffle afterwards.

        # Compute split sizes
        total_size = len(dataset)
        train_size = int(preprocessing_params["train_split"] * total_size)
        val_size = int(preprocessing_params["val_split"] * total_size)
        test_size = total_size - train_size - val_size  # Ensure consistency
        indices = list(range(len(dataset)))

        # Split dataset
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]

        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)
        test_dataset = Subset(dataset, test_indices)
    
    else:
        raise ValueError(f"Unsupported split method: {split_method}")
    
    torch.save(train_dataset, output_folder_artifacts+f"train_dataset.pth")
    torch.save(val_dataset, output_folder_artifacts+f"val_dataset.pth")
    torch.save(test_dataset, output_folder_artifacts+f"test_dataset.pth")

    print(f"Datasets saved in {output_folder_artifacts}")