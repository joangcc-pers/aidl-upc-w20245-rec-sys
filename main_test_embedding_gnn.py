import torch
from torch.utils.data import DataLoader
from scripts.preprocessing_scripts.node_embedding import NodeEmbedding
from scripts.preprocessing_scripts.session_graph_embeddings_dataset import SessionGraphEmbeddingsDataset
from scripts.collate_fn import collate_fn

#########################################################################################
#### SCRIPT FOR DEBUGGING ONLY: DATASET INSTANTIATION WITH EMBEDDINGS AND DATALOADER ####
#########################################################################################
###################################################################
#### CODE FOR DEVELOPMENT AND DEBUGGING ONLY: WORK IN PROGRESS ####
###################################################################

# Configuration
# EMBEDDING_DIM = 64
# NUM_CATEGORIES = 100
# NUM_SUB_CATEGORIES = 50
# NUM_ELEMENTS = 200
# NUM_EVENT_TYPES = 10
FOLDER_PATH = "data/raw/"

if __name__ == "__main__":
    # Initialize NodeEmbedding model
    # embedding_model = NodeEmbedding(NUM_CATEGORIES, NUM_SUB_CATEGORIES, NUM_ELEMENTS, NUM_EVENT_TYPES, EMBEDDING_DIM)

    # Initialize dataset and dataloader
    dataset = SessionGraphEmbeddingsDataset(folder_path=FOLDER_PATH,
                                            start_month='2019-10',
                                            end_month='2019-10'
                                            # , embedding_model
                                            , test_sessions_first_n=10000
                                            , embedding_dim=64
                                            )
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    # Iterate through batches
    for batch in dataloader:
        print(batch)
        break
