from torch.utils.data import DataLoader
from scripts.preprocessing_scripts.node_embedding import NodeEmbedding
from scripts.preprocessing_scripts.session_graph_embeddings_dataset import SessionGraphEmbeddingsDataset
from scripts.collate_fn import collate_fn


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
                                            # embedding_dim=64,
                                            limit_to_view_event=preprocessing_params.get("limit_to_view_event"),
                                            drop_listwise_nulls=preprocessing_params.get("drop_listwise_nulls"),
                                            min_products_per_session=preprocessing_params.get("min_products_per_session")
                                            )
    
    #DEBUGGING:
    # Check shape of one sample
    # Get first sample
    sample = dataset[0]

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

    dataloader = DataLoader(dataset,
                            batch_size=preprocessing_params.get("batch_size"),
                            shuffle=preprocessing_params.get("shuffle"),
                            collate_fn=collate_fn
                            )
    
    for batch in dataloader:
        print(batch)  # Prints a summary of the batch
        break  # Stop after the first batch

    # Inspect batch components
    if hasattr(batch, "product_id_remapped"):
        print(f"Batch node features shape: {batch.product_id_remapped.shape}")
    if hasattr(batch, "category"):
        print(f"Batch category shape: {batch.category.shape}")
    if hasattr(batch, "sub_category"):
        print(f"Batch sub_category shape: {batch.sub_category.shape}")
    if hasattr(batch, "element"):
        print(f"Batch element shape: {batch.element.shape}")
    if hasattr(batch, "brand"):
        print(f"Batch brand shape: {batch.brand.shape}")
    if hasattr(batch, "price_tensor"):
        print(f"Batch price tensor shape: {batch.price_tensor.shape}")
    if hasattr(batch, "edge_index"):
        print(f"Batch edge index shape: {batch.edge_index.shape}")
    if hasattr(batch, "y"):
        print(f"Batch target shape: {batch.y.shape}")
    if hasattr(batch, "batch"):
        print(f"Batch mapping shape: {batch.batch.shape}")  # Shows which nodes belong to which graph


    return dataloader