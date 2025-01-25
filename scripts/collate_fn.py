from torch_geometric.data import Batch

def collate_fn(batch):
    """Custom collate function for PyTorch Geometric batching."""
    return Batch.from_data_list(batch)
