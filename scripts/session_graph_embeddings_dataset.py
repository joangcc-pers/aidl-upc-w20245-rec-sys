import os
import pandas as pd
import torch

from torch.utils.data import Dataset
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder

from utils.csv_files_enum import CsvFilesEnum

class SessionGraphEmbeddingsDataset(Dataset):
    def __init__(self, folder_path, embedding_model, start_month, end_month, transform=None):
        """
        Dataset for session-based recommendation using Graph Neural Networks.
        Args:
            folder_path (str): Path to the folder containing session CSVs.
            embedding_model (NodeEmbedding): Pre-trained embedding model for categorical features.
            start_month (str): Start month in 'YYYY-MM' format.
            end_month (str): End month in 'YYYY-MM' format.
            transform (callable, optional): Transform function for data augmentation.
        """
        print("[INFO] Initializing SessionGraphDataset...")

        # Step 1: Load CSV files

        # Validate and transform input dates using CsvFilesEnum
        try:
            start_csv = CsvFilesEnum.from_date(start_month)
            end_csv = CsvFilesEnum.from_date(end_month)
        except ValueError as e:
            raise ValueError(f"Error in date conversion: {e}")

        # Convert CSV filenames back to 'YYYY-MMM' format for comparison
        start_key = start_csv.replace(".csv", "")
        end_key = end_csv.replace(".csv", "")


        print("[INFO] Loading CSV files from folder:", folder_path)
        csv_files = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.endswith('.csv') and start_key <= f[:7] <= end_key
        ]

        if not csv_files:
            raise FileNotFoundError(
                f"No CSV files found in the specified date range: {start_month} to {end_month}.\n"
                f"Expected filenames: {start_csv} to {end_csv}."
            )


        self.data = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
        print(f"[INFO] Loaded {len(self.data)} rows of data from {len(csv_files)} CSV files.")

        # Step 2: Parse `category_code` into hierarchical features
        print("[INFO] Parsing category hierarchy...")
        # Replace NaN with a default value before splitting
        self.data['category_code'] = self.data['category_code'].fillna('unknown.unknown.unknown')

        # Handle "unknown.unknown.unknown" directly without splitting
        unknown_mask = self.data['category_code'] == 'unknown.unknown.unknown'
        self.data.loc[unknown_mask, ['category', 'sub_category', 'element']] = ['unknown', 'unknown', 'unknown']

        # Split only the non-"unknown" rows
        self.data.loc[~unknown_mask, ['category', 'sub_category', 'element']] = self.data.loc[~unknown_mask, 'category_code'] \
            .str.split('.', expand=True, n=3)
        print("[INFO] Parsed category hierarchy into 'category', 'sub_category', and 'element' columns.")

        # Step 3: Sort by session and timestamp for sequential modeling
        print("[INFO] Sorting data by 'user_session' and 'event_time'...")
        self.data.sort_values(by=['user_session', 'event_time'], inplace=True)
        print("[INFO] Data sorted.")

       # Step 4: Limit `self.data` to only the first 1000 unique sessions
        print("[INFO:TEST_MODE] Limiting data to the first 1000 sessions...")
        first_1000_sessions = self.data['user_session'].unique()[:1000]
        self.data = self.data[self.data['user_session'].isin(first_1000_sessions)]
        print(f"[INFO:TEST_MODE] Data limited to {len(self.data)} rows from the first 1000 sessions.")

        # Step 5: Encode categorical columns
        print("[INFO] Encoding categorical columns...")
        le = LabelEncoder()

        # Fit and transform the categories
        self.data['category'] = le.fit_transform(self.data['category'])
        self.data['sub_category'] = le.fit_transform(self.data['sub_category'])
        self.data['element'] = le.fit_transform(self.data['element'])
        self.data['event_type'] = le.fit_transform(self.data['event_type'])
        print("[INFO] Categorical columns encoded.")

        # Step 6: Extract unique sessions (limited to first 1000)
        print("[INFO] Extracting unique sessions (limited to first 1000)...")
        self.sessions = self.data['user_session'].unique()[:1000]  # Limit to first 1000 sessions
        print(f"[INFO] Found {len(self.sessions)} unique sessions.")

        # Step 7: Declare embeddings model
        print("[INFO] Declaring embeddings for categorical features...")
        self.embedding_model = embedding_model
        print("[INFO] Declared embeddings.")

        self.transform = transform
        print("[INFO] SessionGraphDataset initialization complete.")

    def __len__(self):
        """
        Returns the number of sessions in the dataset.
        """
        return len(self.sessions)
    
    def __getitem__(self, idx):
        """
        Fetches a session and constructs a graph object.
        Returns:
            PyTorch Geometric Data object
        """
        session_id = self.sessions[idx]
        session_data = self.data[self.data['user_session'] == session_id]

        # Pass data through the embedding model during the forward pass
        category_embeddings = self.embedding_model.generate_embeddings(
            torch.tensor(session_data['category'].values, dtype=torch.long),
            torch.tensor(session_data['sub_category'].values, dtype=torch.long),
            torch.tensor(session_data['element'].values, dtype=torch.long),
            torch.tensor(session_data['event_type'].values, dtype=torch.long),
        )
        # Directly pass the embeddings to the graph creation function
        return self._get_graph(session_data, session_id, category_embeddings)

    def _get_graph(self, session_data, session_id, category_embeddings):
        """
        Converts session interactions into a graph.
        Args:
            session_data: Filtered session DataFrame for a specific session.
            session_id: The ID of the session being processed.
            category_embeddings: Precomputed embeddings for the session data.
        """
        print(f"[DEBUG] Building graph for session: {session_id}")

        # Map product IDs to node indices
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

        if not edges:  # Handle empty sessions
            edges.append([0, 0])

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        # Aggregate embeddings by node
        category_embeddings = category_embeddings[:len(unique_products)]  # Match number of unique products
        price_tensor = torch.tensor(
            session_data.groupby('product_id')['price'].first().values, dtype=torch.float
        ).unsqueeze(1)

        # Ensure tensor sizes match
        assert category_embeddings.size(0) == price_tensor.size(0), "Node counts do not match!"

        # Concatenate embeddings and price
        x = torch.cat([category_embeddings, price_tensor], dim=1)

        # PyG graph object
        graph = Data(x=x, edge_index=edge_index, session_id=session_id)

        if self.transform:
            graph = self.transform(graph)

        print(f"[DEBUG] Graph for session {session_id} created with {x.size(0)} nodes and {edge_index.size(1)} edges.")
        return graph


