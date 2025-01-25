import os
import argparse
import yaml
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data, Batch
from sklearn.preprocessing import LabelEncoder
import torch.nn.functional as F
from torch.optim import SGD
import matplotlib.pyplot as plt
import numpy as np

# ===== CsvFilesEnum =====
class CsvFilesEnum:
    @staticmethod
    def from_date(date_str: str) -> str:
        month_map = {
            "01": "Jan", "02": "Feb", "03": "Mar", "04": "Apr", "05": "May", "06": "Jun",
            "07": "Jul", "08": "Aug", "09": "Sep", "10": "Oct", "11": "Nov", "12": "Dec"
        }
        if not isinstance(date_str, str) or len(date_str) != 7 or date_str[4] != "-" or not date_str[:4].isdigit() or not date_str[5:].isdigit():
            raise ValueError("Invalid date format. Expected format is 'YYYY-MM'.")
        year, month = date_str.split("-")
        if month not in month_map:
            raise ValueError(f"Invalid month '{month}'. Expected values are between '01' and '12'.")
        return f"{year}-{month_map[month]}.csv"

# ===== NodeEmbedding =====
class NodeEmbedding(torch.nn.Module):
    def __init__(self, num_categories, num_sub_categories, num_elements, num_event_types, embedding_dim):
        super(NodeEmbedding, self).__init__()
        self.category_embedding = torch.nn.Embedding(num_categories, embedding_dim)
        self.sub_category_embedding = torch.nn.Embedding(num_sub_categories, embedding_dim)
        self.element_embedding = torch.nn.Embedding(num_elements, embedding_dim)
        self.event_type_embedding = torch.nn.Embedding(num_event_types, embedding_dim)

    def generate_embeddings(self, categories, sub_categories, elements, event_types):
        category_emb = self.category_embedding(categories)
        sub_category_emb = self.sub_category_embedding(sub_categories)
        element_emb = self.element_embedding(elements)
        event_type_emb = self.event_type_embedding(event_types)
        return torch.cat([category_emb, sub_category_emb, element_emb, event_type_emb], dim=1)

# ===== Collate Function =====
def collate_fn(batch):
    return Batch.from_data_list(batch)

# ===== Dataset =====
class SessionGraphEmbeddingsDataset(Dataset):
    def __init__(self, folder_path, start_month, end_month, embedding_dim):
        print("[INFO] Initializing SessionGraphDataset...")
        start_csv = CsvFilesEnum.from_date(start_month)
        end_csv = CsvFilesEnum.from_date(end_month)
        start_key, end_key = start_csv.replace(".csv", ""), end_csv.replace(".csv", "")
        csv_files = [
            os.path.join(folder_path, f) for f in os.listdir(folder_path)
            if f.endswith('.csv') and start_key <= f.replace(".csv", "") <= end_key
        ]

        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in date range {start_month} to {end_month}.")

        self.data = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
        self.data['category_code'] = self.data['category_code'].fillna('unknown.unknown.unknown')
        split_data = self.data['category_code'].str.split('.', expand=True).reindex(columns=[0, 1, 2])
        self.data[['category', 'sub_category', 'element']] = split_data.fillna('unknown')
        self.data.sort_values(by=['user_session', 'event_time'], inplace=True)

        le = LabelEncoder()
        self.data['category'] = le.fit_transform(self.data['category'])
        self.data['sub_category'] = le.fit_transform(self.data['sub_category'])
        self.data['element'] = le.fit_transform(self.data['element'])
        self.data['event_type'] = le.fit_transform(self.data['event_type'])

        self.sessions = self.data['user_session'].unique()
        self.embedding_model = NodeEmbedding(
            num_categories=self.data['category'].nunique(),
            num_sub_categories=self.data['sub_category'].nunique(),
            num_elements=self.data['element'].nunique(),
            num_event_types=self.data['event_type'].nunique(),
            embedding_dim=embedding_dim
        )

    def __len__(self):
        return len(self.sessions)

    def __getitem__(self, idx):
        session_id = self.sessions[idx]
        session_data = self.data[self.data['user_session'] == session_id]
        category_embeddings = self.embedding_model.generate_embeddings(
            torch.tensor(session_data['category'].values, dtype=torch.long),
            torch.tensor(session_data['sub_category'].values, dtype=torch.long),
            torch.tensor(session_data['element'].values, dtype=torch.long),
            torch.tensor(session_data['event_type'].values, dtype=torch.long)
        )
        return self._get_graph(session_data, session_id, category_embeddings)

    def _get_graph(self, session_data, session_id, category_embeddings):
        product_ids = session_data['product_id'].astype('category')
        unique_products = product_ids.cat.categories
        node_map = {pid: i for i, pid in enumerate(unique_products)}

        edges = []
        for i in range(len(session_data) - 1):
            src, dst = node_map[session_data.iloc[i]['product_id']], node_map[session_data.iloc[i + 1]['product_id']]
            edges.append([src, dst])
            edges.append([dst, src])

        edge_index = torch.tensor(edges if edges else [[0, 0]], dtype=torch.long).t().contiguous()
        category_embeddings = category_embeddings[:len(unique_products)]
        price_tensor = torch.tensor(session_data.groupby('product_id')['price'].first().values, dtype=torch.float).unsqueeze(1)
        x = torch.cat([category_embeddings, price_tensor], dim=1)
        graph = Data(x=x, edge_index=edge_index, session_id=session_id)
        return graph

# ===== Model =====
class GraphModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GraphModel, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)

    def forward(self, data):
        x = F.relu(self.fc1(data.x))
        x = self.fc2(x)
        return x

# ===== Training ===== EJ, SGD, CAMBIAR EN EL MODELO QUE USEMOS
def train_model(model, dataloader, epochs):
    #optimizer = SGD(model.parameters(), lr=0.01)
    #for epoch in range(epochs):
        #for batch in dataloader:
            #optimizer.zero_grad()
           # output = model(batch)
           # loss = F.mse_loss(output, torch.zeros_like(output))  # Placeholder for loss calculation
           # loss.backward()
          #  optimizer.step()
      #  print(f"Epoch {epoch + 1}/{epochs} completed.")

# ===== Evaluation =====
def evaluate_model(model, dataloader):
    # Placeholder for evaluation logic
   # model.eval()
   # total_loss = 0
  #  with torch.no_grad():
      #  for batch in dataloader:
         #   output = model(batch)
           # loss = F.mse_loss(output, torch.zeros_like(output))  # Placeholder for loss calculation
          #  total_loss += loss.item()
   # avg_loss = total_loss / len(dataloader)
    #print(f"Evaluation completed. Average loss: {avg_loss}")

# ===== Visualization =====
def visualize_model(model, dataloader):
    # Placeholder for visualization logic
    print("[INFO] Visualizing model...")
    embeddings = []
    with torch.no_grad():
        for batch in dataloader:
            output = model(batch)
            embeddings.append(output.cpu().numpy())

    embeddings = np.concatenate(embeddings, axis=0)
    plt.scatter(embeddings[:, 0], embeddings[:, 1])
    plt.title("Embedding Visualization")
    plt.xlabel("Embedding Dimension 1")
    plt.ylabel("Embedding Dimension 2")
    plt.show()

# ===== Main =====
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RecSys experiments.")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--experiment", type=str, required=True, help="Experiment name")
    parser.add_argument("--task", nargs="+", required=True, help="Tasks to perform (e.g., clean train evaluate visualize)")
    args = parser.parse_args()

    # Validate task order
    valid_order = ["clean", "train", "evaluate", "visualize"]
    validate_task_order(args.task, valid_order)

    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    experiment_config = config["experiments"][args.experiment]

    # Run tasks dynamically
    dataset, dataloader = None, None
    model = None
    if "clean" in args.task:
        dataset = SessionGraphEmbeddingsDataset(
            folder_path=experiment_config["data_params"]["raw_input_path"],
            start_month=experiment_config["data_params"]["start_month"],
            end_month=experiment_config["data_params"]["end_month"],
            embedding_dim=experiment_config["model_params"]["embedding_dim"]
        )
        dataloader = DataLoader(dataset, batch_size=experiment_config["data_params"]["batch_size"], collate_fn=collate_fn)

    if "train" in args.task and dataset:
        model = NodeEmbedding(
            num_categories=dataset.data['category'].nunique(),
            num_sub_categories=dataset.data['sub_category'].nunique(),
            num_elements=dataset.data['element'].nunique(),
            num_event_types=dataset.data['event_type'].nunique(),
            embedding_dim=experiment_config["model_params"]["embedding_dim"]
        )
        train_model(model, dataloader, epochs=experiment_config["train_params"]["epochs"])

    if "evaluate" in args.task and model:
        evaluate_model(model, dataloader)

    if "visualize" in args.task and model:
        visualize_model(model, dataloader)
