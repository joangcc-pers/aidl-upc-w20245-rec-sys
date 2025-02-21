import os
import pandas as pd
import torch

import json

import numpy as np

from torch.utils.data import Dataset
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder

from utils.csv_files_enum import CsvFilesEnum

#from scripts.preprocessing_scripts.node_embedding import NodeEmbedding
from sklearn.preprocessing import OneHotEncoder

class SessionGraphOneHotDataset(Dataset):
    def __init__(self,
                 folder_path,
                 output_folder_artifacts,
                 start_month,
                 end_month,
                 transform=None,
                 test_sessions_first_n=None,
                 limit_to_view_event=False,
                 drop_listwise_nulls=False,
                 min_products_per_session=3,
                 normalization_method="min_max"
                 ):
        """
        Dataset for session-based recommendation using Graph Neural Networks.
        Args:
            folder_path (str): Path to the folder containing session CSVs.
            output_folder_artifacts (str): Path to the folder where the artifacts will be saved.
            start_month (str): Start month in 'YYYY-MM' format.
            end_month (str): End month in 'YYYY-MM' format.
            transform (callable, optional): Transform function for data augmentation.
            test_sessions_first_n (int, optional): Limit the dataset to the first n sessions for testing/debugging purposes.
            limit_to_view_event (bool, optional): Limit the dataset to 'view' events only.
            drop_listwise_nulls (bool, optional): Drop rows with nulls in 'brand', 'category_code' or 'price'
            min_products_per_session (int, optional): Minimum number of unique products per session.
            normalization_method (str, optional): Method for normalizing the price column. Options: 'min_max', 'z_score'.
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
        csv_files = []
        for f in os.listdir(folder_path):
            if f.endswith('.csv'):
                # Extract the portion of the filename to compare
                filename_without_extension = os.path.splitext(f)[0]
                key = filename_without_extension.split("/")[-1]  # Handle forward slash or use full name
                if start_key <= key <= end_key:
                    csv_files.append(os.path.join(folder_path, f))

        if not csv_files:
            raise FileNotFoundError(
                f"No CSV files found in the specified date range: {start_month} to {end_month}.\n"
                f"Expected filenames: {start_csv} to {end_csv}."
            )

        # Load CSV files into a single DataFrame
        self.data = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
        print(f"[INFO] Loaded {len(self.data)} rows of data from {len(csv_files)} CSV files.")
        

        # Step 2: Normalize the price column
        # Extract the price column as a NumPy array for faster computations
        print(f"[INFO] Normalizing price column using {normalization_method}...")
        price = self.data["price"].values  

        if normalization_method == "min_max":
            min_val = np.min(price)
            max_val = np.max(price)
            range_val = max_val - min_val
            if range_val != 0:  # Avoid division by zero
                self.data["price"] = (price - min_val) / range_val
            else:
                self.data["price"] = 0  # If all values are the same, set to 0
        elif normalization_method == "z_score":
            mean_val = np.mean(price)
            std_val = np.std(price)
            if std_val != 0:  # Avoid division by zero
                self.data["price"] = (price - mean_val) / std_val
            else:
                self.data["price"] = 0  # If all values are the same, set to 0

        # Step 3: Limit to event_type = view
        if limit_to_view_event == True:
            self.data = self.data[self.data['event_type'] == 'view'].reset_index(drop=True)
            print(f"[INFO] Data limited to {len(self.data)} rows with event_type = 'view'.")


        # Step 4: Limit to no nulls in brand, category_code and price
        if drop_listwise_nulls == True:
            self.data = self.data.dropna(subset=['brand', 'category_code', 'price']).reset_index(drop=True)
            print(f"[INFO] Data limited to {len(self.data)} rows with no nulls in 'brand', 'category_code' and 'price'.")

        # Group by 'user_session' and count the number of unique 'product_id' values for each session
        product_counts_per_session = self.data.groupby('user_session')['product_id'].nunique()

        # Filter sessions where there are two or more unique products
        valid_sessions = product_counts_per_session[product_counts_per_session >= min_products_per_session].index

        # Keep only the rows where the 'user_session' is in the valid sessions
        self.data = self.data[self.data['user_session'].isin(valid_sessions)].reset_index(drop=True)

        print(f"[INFO] Data limited to {len(self.data)} rows with {min_products_per_session} or more unique products per session.")

        # Step 5 [Debugging]: Limit `self.data` to only the first k unique sessions of the concatenated CSVs

        if test_sessions_first_n:
            print(f"[INFO:TEST_MODE] Limiting data to the first {str(test_sessions_first_n)} sessions...")
            first_n_sessions = self.data['user_session'].unique()[:test_sessions_first_n]
            self.data = self.data[self.data['user_session'].isin(first_n_sessions)].reset_index(drop=True)
            print(f"[INFO:TEST_MODE] Data limited to {len(self.data)} rows from the first {str(test_sessions_first_n)} sessions.")


        # Agrupar por user_session y contar el número de productos únicos en cada sesión con al menos 3 eventos "view"
        
        product_counts_per_session = self.data.groupby('user_session')['product_id'].nunique()

        #Raise an error if product_counts_per_session is equal to 1 in any instance
        if (product_counts_per_session == 1).any():
            raise ValueError("There are sessions with only 1 unique product. Please filter them out.")


        # Step 5: Parse `category_code` into hierarchical features
        print("[INFO] Parsing category hierarchy...")

        # Create a mask for rows where `category_code` is NaN
        unknown_mask = self.data['category_code'].isna()

        # Replace NaN values using the mask
        self.data.loc[unknown_mask, 'category_code'] = 'unknown.unknown.unknown'

        # Split the `category_code` dynamically into up to 3 parts
        split_data = self.data['category_code'].str.split('.', expand=True).reindex(columns=[0, 1, 2])

        # Assign split values back to the respective columns
        self.data[['category', 'sub_category', 'element']] = split_data.fillna('unknown')

        # Debugging information
        print(self.data[['category', 'sub_category', 'element']])

        print("[INFO] Parsed category hierarchy into 'category', 'sub_category', and 'element' columns.")

        # Debugging information
        print("Category after parsing")
        print(self.data['category'].unique())
        print(self.data['sub_category'].unique())
        print(self.data['element'].unique())

        # Step 6: do a linear transformation of the feature product_id to a new feature product_id_remapped

        # Create a new column with the remapped product_id
        self.data['product_id_remapped'] = self.data['product_id'].astype('category')
        self.data['product_id_remapped'] = self.data['product_id_remapped'].cat.codes


        # Step 7: Sort by session and timestamp for sequential modeling
        print("[INFO] Sorting data by 'user_session' and 'event_time'...")
        self.data.sort_values(by=['user_session', 'event_time'], inplace=True)
        print("[INFO] Data sorted.")

        # Step 8: Calculate number of unique occurrences for each column
        num_categories = self.data['category'].nunique()
        num_sub_categories = self.data['sub_category'].nunique()
        num_elements = self.data['element'].nunique()
        num_brands = self.data['brand'].nunique()
        num_items = self.data['product_id_remapped'].nunique()

        # Save the unique values of each column in a JSON file to pass them to NodeEmbedding
        num_values_for_node_encoding = {
            'num_categories': num_categories,
            'num_sub_categories': num_sub_categories,
            'num_elements': num_elements,
            'num_brands': num_brands,
            'num_items': num_items
        }

        #Create a new json file with the unique values of each column
        os.makedirs(output_folder_artifacts, exist_ok=True)
        export_json_path = os.path.join(output_folder_artifacts, 'num_values_for_node_embedding.json')
        with open(export_json_path, 'w') as f:
            json.dump(num_values_for_node_encoding, f)

        #TODO: retornar els valors únics de cada columna per a passar-els al NodeEmbedding

        # Debugging information
        print(f"[DEBUG] Unique categories count: {num_categories}")
        # print(f"[DEBUG] Unique categories: {self.data['category'].unique()}")
        print(f"[DEBUG] Unique sub-categories count: {num_sub_categories}")
        # print(f"[DEBUG] Unique sub-categories: {self.data['sub_category'].unique()}")
        print(f"[DEBUG] Unique elements count: {num_elements}")
        # print(f"[DEBUG] Unique elements: {self.data['element'].unique()}")
        print(f"[DEBUG] Unique brands count: {num_brands}")
        # print(f"[DEBUG] Unique brand names: {self.data['brand'].unique()}")

        # Fit and transform the categories
        self.categorical_columns = ['category', 'sub_category', 'element', 'brand']
        self.data = self.data.drop(columns=['category_code', 'category_id'])
        self.encoder = OneHotEncoder(sparse_output=False)
        # Fit and transform all categorical columns at once
        categorical_columns = ['category', 'sub_category', 'element', 'brand']
        encoded_features = self.encoder.fit_transform(self.data[categorical_columns])
        
        
        
        # Convert to DataFrame to maintain column names (optional but helpful for debugging)
        feature_names = self.encoder.get_feature_names_out(categorical_columns)
        encoded_df = pd.DataFrame(encoded_features, columns=feature_names, index=self.data.index)
        
        # Replace original categorical columns with encoded ones
        self.data = self.data.drop(columns=categorical_columns)
        self.data = pd.concat([self.data, encoded_df], axis=1)
        

        print("\n[DEBUG] Brand encoding check:")
        brand_columns = [col for col in self.data.columns if col.startswith('brand_')]
        print(f"Number of brand columns created: {len(brand_columns)}")
        print(f"Sample brand columns: {brand_columns[:5]}")
        print(f"Sum of brand values: {self.data[brand_columns].sum().sum()}")

        
        
        print("[INFO] Categorical columns encoded.")
        print(f"[DEBUG] Encoded features shape: {encoded_features.shape}")



        # Step 9: Extract unique sessions
        print("[INFO] Extracting unique sessions...")
        self.sessions = self.data['user_session'].unique()
        print(f"[INFO] Found {len(self.sessions)} unique sessions.")

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
        


        # Get one-hot encoded features by selecting columns with their respective prefixes
        encoded_labels_dict = {
            'category': torch.tensor(
                session_data[[col for col in session_data.columns if col.startswith('category_') and col != 'category_code']].values, 
                dtype=torch.float
            ),
            'sub_category': torch.tensor(
                session_data[[col for col in session_data.columns if col.startswith('sub_category_')]].values,
                dtype=torch.float
            ),
            'element': torch.tensor(
                session_data[[col for col in session_data.columns if col.startswith('element_')]].values,
                dtype=torch.float
            ),
            'brand': torch.tensor(
                session_data[[col for col in session_data.columns if col.startswith('brand_')]].values,
                dtype=torch.float
            )
        }
        # Directly pass the encoded labels to the graph creation function
        return self._get_graph(session_data, session_id, encoded_labels_dict)

    def _get_graph(self, session_data, session_id, encoded_labels_dict):
        """
        Converts session interactions into a graph.
        Args:
            session_data: Filtered session DataFrame for a specific session.
            session_id: The ID of the session being processed.
            encoded_labels_dict: Encoded labels for the session data.
        """
        # print(f"[DEBUG] Building graph for session: {session_id}")

        # Map product IDs to node indices
        session_data_without_last = session_data.iloc[:-1]
        #node_ids = session_data_without_last['product_id_remapped'].unique()
        #node_map = {pid: i for i, pid in enumerate(node_ids)}


        # Building edge list (temporal order + bidirectional edges)
        session_data_without_last = session_data.iloc[:-1]
        product_ids = session_data_without_last['product_id_remapped'].unique()
        node_map = {pid: i for i, pid in enumerate(product_ids)}

        # Building edge list (temporal order + bidirectional edges)
        edges = []
        for i in range(len(session_data_without_last) - 1):
            src = node_map[session_data_without_last.iloc[i]['product_id_remapped']]
            dst = node_map[session_data_without_last.iloc[i + 1]['product_id_remapped']]
            edges.append([src, dst])
            edges.append([dst, src])

        if not edges:  # Handle empty sessions
            edges.append([0, 0])

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            # Quitar el último evento de la sesión de `price`
        price_tensor = torch.tensor(
            session_data['price'][:-1].values, dtype=torch.float
        ).unsqueeze(1)

        # Target (the last product ID to predict)
        target_product_id_global = session_data.iloc[-1]['product_id_remapped']
        y = torch.tensor(target_product_id_global, dtype=torch.long)

        # PyG graph object
        graph = Data(price_tensor=price_tensor,
                    #  product_id_global_tensor=product_id_global_tensor,
                     edge_index=edge_index,
                     y=y,
                     session_id=session_id,
                     category=encoded_labels_dict['category'][:-1],
                     sub_category=encoded_labels_dict['sub_category'][:-1],
                     element=encoded_labels_dict['element'][:-1],
                     brand=encoded_labels_dict['brand'][:-1],
                     product_id_remapped=torch.tensor(session_data['product_id_remapped'][:-1].values, dtype=torch.long),
                     num_nodes = len(session_data_without_last)
                     )

        if self.transform:
            graph = self.transform(graph)

        # print(f"[DEBUG] Graph for session {session_id} created with {x.size(0)} nodes and {edge_index.size(1)} edges.")
        return graph
    