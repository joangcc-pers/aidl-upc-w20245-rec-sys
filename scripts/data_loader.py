from torch_geometric.data import Data
import torch

def create_data(categories, sub_categories, elements, edge_index):
    """
    Creates a `Data` object for graph modeling.

    Args:
        categories (torch.Tensor): Tensor of category IDs.
        sub_categories (torch.Tensor): Tensor of sub-category IDs.
        elements (torch.Tensor): Tensor of element IDs.
        edge_index (torch.Tensor): Edge connections for the graph.

    Returns:
        Data: PyTorch Geometric data object.
    """
    return Data(
        category=categories,
        sub_category=sub_categories,
        element=elements,
        edge_index=edge_index
    )
