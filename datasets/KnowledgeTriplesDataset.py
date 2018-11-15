from torch.utils.data import Dataset, DataLoader
import os
import torch
from pandas import pd


class KnowledgeTriplesDataset(Dataset):
    """ Knowledge Triples (s,r,o) Dataset """

    def __init__(self, csv_file, root_dir):
        """
        Args:
        :param csv_file: dataset filename
        :param root_dir: dataset filepath
        """
        