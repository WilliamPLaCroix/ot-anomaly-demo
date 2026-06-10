from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import kagglehub
from kagglehub import KaggleDatasetAdapter
import os

# os.environ["KAGGLEHUB_CACHE"] = "../data"

def prepare_dataset(df: pd.DataFrame, debug: bool = False):
    # Load and preprocess dataset

    df.columns = df.columns.str.strip()
    if debug:
        df = df.head(1000)
    labels = df['Normal/Attack']
    timestamps = df['Timestamp']
    features = df.drop(columns=['Normal/Attack', 'Timestamp'])
    features = (features - features.min()) / (features.max() - features.min())
    features = features.fillna(-1)
    
    return features, labels, timestamps

class CustomDataset(Dataset):
    """
    data_iter = iter(train_dataloader)
    features, labels, timestamps = next(data_iter)
    """
    def __init__(self, features, labels, timestamps):
        self.features = features
        self.labels = labels
        self.timestamps = timestamps

    def __getitem__(self, index):
        features = self.features.iloc[index].to_numpy()
        label = self.labels.iloc[index]
        timestamp = self.timestamps.iloc[index]
        return features, label, timestamp

    def __len__(self):
        return len(self.features)

def create_dataloader(split: str = "normal", batch_size: int = 32, debug: bool = False) -> DataLoader:
    # print current working directory
    df = pd.read_csv(f"data/{split}.csv")
    # file_path = split + ".csv"
    # # Load the latest version
    # df = kagglehub.load_dataset(
    # KaggleDatasetAdapter.PANDAS,
    # "vishala28/swat-dataset-secure-water-treatment-system",
    # file_path,
    # )
    features, labels, timestamps = prepare_dataset(df, debug=debug)
    dataset = CustomDataset(features, labels, timestamps)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
