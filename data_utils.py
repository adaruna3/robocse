from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from os.path import abspath,dirname
from logging.viz_utils import tp
import datasets


class KnowledgeTriplesDataset(Dataset):
    """ Knowledge Triples (s,r,o) Dataset """

    def __init__(self,
                 dataset_name,
                 experiment_name,
                 negative_sampling_ratio,neg_sampling_method):
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
        self.neg_ratio = negative_sampling_ratio
        self.neg_method = neg_sampling_method

    def load_id_map(self,csv_file):
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

    def load_triples(self,csv_file):
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

    def __getitem__(self,idx):
        """
        :param idx: indexes of triples to return
        :return: training triples sample
        """
        samples = np.reshape(np.append(self.sros[idx,:],[1]),(1,4))
        for i in xrange(self.neg_ratio):
            samples = np.append(samples,
                                self.sample_negative(self.sros[idx,:]),
                                axis=0)
        return samples

    def sample_negative(self,triple):
        """
        creates three negative samples with labels
        :param triple: triple used for generating negative samples
        :return: three negative samples
        """
        s,r,o = triple.tolist()
        ss = self.neg_s(r,o)
        rr = self.neg_r(s,o)
        oo = self.neg_o(s,r)
        negatives = [[ss,r,o,-1]]
        negatives.append([s,rr,o,-1])
        negatives.append([s,r,oo,-1])
        return np.asarray(negatives)

    def neg_s(self,rel,obj):
        if self.neg_method == 'random':
            return np.random.randint(0,len(self.i2e))

    def neg_r(self,subj,obj):
        if self.neg_method == 'random':
            return np.random.randint(0,len(self.i2r))

    def neg_o(self,subj,rel):
        if self.neg_method == 'random':
            return np.random.randint(0,len(self.i2e))


if __name__ == "__main__":
    # creates triples dataset
    dataset = KnowledgeTriplesDataset("demo","train",1,'random')
    # loads triples for training
    batch_size = 2
    num_threads = 1
    dataset_loader = DataLoader(dataset,batch_size=batch_size,shuffle=False,
                                num_workers=num_threads)
    # loop through data printing batch number and samples
    for idx_b, batch in enumerate(dataset_loader):
        print idx_b
        print batch