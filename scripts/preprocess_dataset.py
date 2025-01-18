import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch

class SessionGraphDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        """
        Dataset for session-based recommendation using Graph Neural Networks.
        Args:
            folder_path (str): Path to the folder containing session CSVs.
            transform (callable, optional): Transform function for data augmentation.
        """
        # Concadenate CSVs fields
        csv_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]
        self.data = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

        # Parsing category_code into category hierarchy
        self.data[['category', 'sub_category', 'element']] = self.data['category_code'] \
            .fillna('unknown.unknown.unknown').str.split('.', expand=True)

        # Sort by session and timestamp for sequential modeling
        self.data.sort_values(by=['user_session', 'event_time'], inplace=True)

        # Unique sessions
        self.sessions = self.data['user_session'].unique()
        self.transform = transform

        # TO encode categorical features (category, event type)
        self.data['category'] = self.data['category'].astype('category').cat.codes
        self.data['sub_category'] = self.data['sub_category'].astype('category').cat.codes
        self.data['element'] = self.data['element'].astype('category').cat.codes
        self.data['event_type'] = self.data['event_type'].astype('category').cat.codes

    def __len__(self):
        return len(self.sessions)

    def __getitem__(self, idx):
        """
        Fetches a session and constructs a graph object.
        Returns:
            PyTorch Geometric Data object
        """
        session_id = self.sessions[idx]
        session_data = self.data[self.data['user_session'] == session_id]

        return self._get_graph(session_data, session_id)

    def _get_graph(self, session_data, session_id):
        """
        Converts session interactions into a graph.
        """
        # Mapping product IDs to node indices
        product_ids = session_data['product_id'].astype('category')
        unique_products = product_ids.cat.categories
        node_map = {pid: i for i, pid in enumerate(unique_products)}

        # Building edge list (temporal order + bidirectional edges)
        edges = []
        for i in range(len(session_data) - 1):
            src = node_map[session_data.iloc[i]['product_id']]
            dst = node_map[session_data.iloc[i + 1]['product_id']]
            edges.append([src, dst])
            edges.append([dst, src])  

        if not edges:  # To handle empty sessions
            edges.append([0, 0])

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        # Node features (category, event type, price)
        x = torch.tensor(
            session_data[['category', 'sub_category', 'element', 'event_type', 'price']].values,
            dtype=torch.float
        )

        #  PyG graph object
        graph = Data(x=x, edge_index=edge_index, session_id=session_id)

        if self.transform:
            graph = self.transform(graph)

        return graph


def collate_fn(batch):
    """Custom collate function for PyTorch Geometric batching."""
    return Batch.from_data_list(batch)


if __name__ == "__main__":
    folder_path = "multicategory"

    dataset = SessionGraphDataset(folder_path)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

    for batch in dataloader:
        print(batch)
        break  

