import torch
from torch.utils.data import Dataset
import pandas as pd
from collections import defaultdict

class NextItemDataset(Dataset):
    def __init__(self, path, mode='gru', max_seq_len=None):
        """
        Args:
            path: ruta al csv
            mode (str): 'graph' or 'gru',
            max_seq_len: Maximum length of sequences for padding.
        """
        self.data = pd.read_csv(path)
        self.mode = mode
        self.max_seq_len = max_seq_len
        
        # Preprocess the data
        self.data['event_time'] = pd.to_datetime(self.data['event_time'])
        self.data.sort_values(by=['user_session', 'event_time'], inplace=True)
        
        self.sessions = self._group_sessions()

    def _group_sessions(self):
        """Groups data by session and creates session sequences."""
        grouped = defaultdict(list)
        for _, row in self.data.iterrows():
            grouped[row['user_session']].append(row['product_id'])
        return list(grouped.values())

    def __len__(self):
        return len(self.sessions)