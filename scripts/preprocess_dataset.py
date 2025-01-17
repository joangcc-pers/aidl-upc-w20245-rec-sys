import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class SessionDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        """
        Custom Dataset for session-based recommendation.
        Args:
            folder_path (str)
            transform (callable, optional): Optional 
        """
        # Concadenation and load of all the fields into the folder
        csv_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]
        self.data = pd.concat([pd.read_csv(f) for f in csv_files]) #reading and merging

        # Preprocessing: parsing the category_code into category, sub-category, and element
        self.data[['category', 'sub_category', 'element']] = self.data['category_code'] \
            .fillna('unknown.unknown.unknown').str.split('.', expand=True)
                # The category_code field is split into three columns: category, sub_category, and element
                # Missing or NaN values are replaced with unknown 

        # Sorting data by session and time for sequential modeling
        self.data.sort_values(by=['user_session', 'event_time'], inplace=True)
        
        # Storing unique sessions
        self.sessions = self.data['user_session'].unique()
        self.transform = transform

        # self.data == preprocessed dataset

    def __len__(self):
        return len(self.sessions)

    def __getitem__(self, idx):
        """
        Get a session's data by index.
        Args:
            idx (int): Index of the session.
        Returns:
            session_data (dict): Containing session info.
        """
        session_id = self.sessions[idx]
        session_data = self.data[self.data['user_session'] == session_id]

        # Extracting features: product_id, category fields, and event type
        items = session_data[['product_id', 'category', 'sub_category', 'element']].values
        event_types = session_data['event_type'].values

        if self.transform:
            items = self.transform(items)

        return {
            'session_id': session_id,
            'items': items,
            'event_types': event_types
        }


if __name__ == "__main__":
    # I changed mine to multicategory and inside there are the two csvs
    folder_path = "multicategory"

    # Create dataset and dataloader
    dataset = SessionDataset(folder_path)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True) #trying with the batch_size

    # Iterate through batches
    for batch in dataloader:
        print(batch)
