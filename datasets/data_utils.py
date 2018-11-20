from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from os.path import abspath,dirname
from viz_utils import tp
import datasets

class KnowledgeTriplesDataset(Dataset):
    """ Knowledge Triples (s,r,o) Dataset """

    def __init__(self, dataset_name, experiment_name):
        """
        :param csv_file: dataset filename
        :param root_dir: dataset filepath
        """
        datasets_fp = abspath(dirname(datasets.__file__)) + '/'
        dataset_fp = datasets_fp + dataset_name
        ent_csv = dataset_fp + '_entities.csv'
        rel_csv = dataset_fp + '_relations.csv'
        train_csv = dataset_fp + '_' + experiment_name + '.csv'
        self.e2i, self.i2e = self.load_id_map(ent_csv)
        self.r2i, self.i2r = self.load_id_map(rel_csv)
        self.sros = self.load_triples(train_csv)

    def load_id_map(self, csv_file):
        """
        :param csv_file: filename of labels
        :return: ID mapping for a set of labels in a file
        """
        try:
            labels = pd.read_csv(csv_file)
        except ValueError as e:
            tp('f','Could not load ' + str(csv_file))
            raise ValueError

        return ({labels.iloc[idx,0]:idx for idx in xrange(labels.size)},
                {idx:labels.iloc[idx,0] for idx in xrange(labels.size)})

    def load_triples(self, csv_file):
        """
        :param csv_file: filename of triples
        :return: loads a set of triples in a file
        """
        try:
            str_triples = pd.read_csv(csv_file)
        except ValueError as e:
            tp('f','Could not load ' + str(csv_file))
            raise ValueError

        num_rows = str_triples.shape[0]
        int_triples = [[self.e2i[str_triples.iloc[idx,0]],
                        self.r2i[str_triples.iloc[idx,1]],
                        self.e2i[str_triples.iloc[idx,2]]]
                       for idx in xrange(num_rows)]
        return np.asarray(int_triples)

    def __len__(self):
        return len(self.sros)

    def __getitem__(self, idx):
        """
        :param idx: indexes of triples to return
        :return: triples
        """
        return self.sros[idx,:]


if __name__ == "__main__":
    # creates triples dataset
    dataset = KnowledgeTriplesDataset("demo","train")
    # loads triples for training
    batch_size = 2
    num_threads = 1
    dataset_loader = DataLoader(dataset,batch_size=batch_size,shuffle=False,
                                num_workers=num_threads)
    # loop through data printing batch number and samples
    for idx_b, batch in enumerate(dataset_loader):
        print idx_b
        print batch