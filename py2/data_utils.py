from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from os.path import abspath,dirname
from robocse_logging.viz_utils import tp
import datasets
from re import split

import pdb


class TripleDataset(Dataset):
    def __init__(self,
                 dataset_name,
                 experiment_name,
                 negative_sampling_ratio=0,
                 neg_sampling_method=None,
                 exclude_tain=0):
        """
        :param csv_file: dataset filename
        :param root_dir: dataset filepath
        """
        datasets_fp = abspath(dirname(datasets.__file__)) + '/'
        dataset_fp = datasets_fp + dataset_name
        ent_csv = dataset_fp + '_entities.csv'
        rel_csv = dataset_fp + '_relations.csv'
        self.e2i, self.i2e = self.load_id_map(ent_csv)
        self.r2i, self.i2r = self.load_id_map(rel_csv)
        self.sros = self.load_triples(dataset_name,experiment_name)

    def load_id_map(self,csv_file):
        """
        :param csv_file: filename of labels
        :return: ID mapping for a set of labels in a file
        """
        try:
            labels = pd.read_csv(csv_file)
        except IOError as e:
            tp('f','Could not load ' + str(csv_file))
            raise IOError

        return ({labels.iloc[idx,0]:idx for idx in xrange(labels.size)},
                {idx:labels.iloc[idx,0] for idx in xrange(labels.size)})

    def load_triples(self,dataset_name,experiment_name):
        """

        :param dataset_name:
        :param experiment_name:
        :return:
        """
        datasets_fp = abspath(dirname(datasets.__file__)) + '/'
        dataset_fp = datasets_fp + dataset_name
        csv_file = dataset_fp + '_' + experiment_name + '.csv'
        try:
            str_triples = pd.read_csv(csv_file)
        except IOError as e:
            tp('f','Could not load ' + str(csv_file))
            raise IOError

        num_rows = str_triples.shape[0]
        int_triples = [[self.e2i[str_triples.iloc[idx,0]],
                        self.r2i[str_triples.iloc[idx,1]],
                        self.e2i[str_triples.iloc[idx,2]]]
                       for idx in xrange(num_rows)]
        return np.asarray(int_triples)

    def __len__(self):
        return len(self.sros)

    def __getitem__(self,idx):
        return self.sros[idx,:]


class TrainDataset(TripleDataset):
    """ Knowledge Triples (s,r,o) Dataset """
    def __init__(self,
                 dataset_name,
                 experiment_name,
                 negative_sampling_ratio=0,
                 neg_sampling_method=None,
                 exclude_tain=0):
        """
        :param csv_file: dataset filename
        :param root_dir: dataset filepath
        """
        super(TrainDataset, self).__init__(dataset_name,
                                           experiment_name)
        self.neg_ratio = negative_sampling_ratio
        self.neg_method = neg_sampling_method

    def __getitem__(self,idx):
        """
        :param idx: indexes of triples to return
        :return: training triples sample (s,r,o,y)
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


class PredictDataset(TripleDataset):
    def __init__(self,
                 dataset_name,
                 experiment_name,
                 negative_sampling_ratio=0,
                 neg_sampling_method=None,
                 exclude_train=0):
        """
        :param dataset_name:
        :param experiment_name:
        """
        super(PredictDataset, self).__init__(dataset_name,
                                             experiment_name)
        self.ground_truth = GT(dataset_name,experiment_name,exclude_train)

    def __getitem__(self,idx):
        """
        :param idx: indexes of triples to return
        :return: training triples sample (s,r,o,y,q,y')
        """
        samples = np.empty((0,6),dtype=int)
        for q_idx in xrange(3):
            true_rank = self.ground_truth.get_ranks(q_idx,self.sros[idx,:])
            sample = np.append(self.sros[idx,:],[1,q_idx,true_rank],axis=0)
            samples = np.append(samples,np.reshape(sample,(1,6)),axis=0)
        return samples


class TriplesCounter(object):
    def __init__(self,dataset_name,experiment_name):
        """
        :param csv_file: dataset filename
        :param root_dir: dataset filepath
        """
        super(TriplesCounter, self).__init__()
        datasets_fp = abspath(dirname(datasets.__file__)) + '/'
        dataset_fp = datasets_fp + dataset_name
        ent_csv = dataset_fp + '_entities.csv'
        rel_csv = dataset_fp + '_relations.csv'
        self.e2i, self.i2e = self.load_id_map(ent_csv)
        self.r2i, self.i2r = self.load_id_map(rel_csv)
        # loads the ground truth data
        counts_shape = (len(self.r2i),len(self.e2i),len(self.e2i))
        self.counts = np.zeros(shape=counts_shape,dtype=np.int64)
        self.load_counts(dataset_name,experiment_name)

    def load_id_map(self,csv_file):
        """
        :param csv_file: filename of labels
        :return: ID mapping for a set of labels in a file
        """
        try:
            labels = pd.read_csv(csv_file)
        except IOError as e:
            tp('f','Could not load ' + str(csv_file))
            raise IOError

        return ({labels.iloc[idx,0]:idx for idx in xrange(labels.size)},
                {idx:labels.iloc[idx,0] for idx in xrange(labels.size)})

    def load_counts(self,dataset_name,experiment_name):
        """
        :param csv_file: filename of triples
        :return: loads a set of triples in a file
        """
        datasets_fp = abspath(dirname(datasets.__file__)) + '/'
        dataset_fp = datasets_fp + dataset_name
        gt_csv_fp = dataset_fp+'_'+experiment_name+'.csv'
        try:  # reads CSV
            str_triples = pd.read_csv(gt_csv_fp)
        except IOError as e:
            tp('f','Could not load ' + str(gt_csv_fp))
            raise IOError
        # loads into memory datastructure
        num_rows = str_triples.shape[0]
        for idx in xrange(num_rows):
            self.counts[self.r2i[str_triples.iloc[idx,1]],
                        self.e2i[str_triples.iloc[idx,0]],
                        self.e2i[str_triples.iloc[idx,2]]] += 1

    def clear_model(self):
        self.counts = np.zeros(shape=self.counts.shape,dtype=np.int64)


class GT(TriplesCounter):
    def __init__(self,dataset_name,experiment_name,exclude_train):
        self.exclude = exclude_train
        super(GT, self).__init__(dataset_name,experiment_name)
        self.clear_model()
        self.load_counts(dataset_name,experiment_name)

    def load_counts(self,dataset_name,experiment_name):
        """
        :param csv_file: filename of triples
        :return: loads a set of triples in a file
        """
        datasets_fp = abspath(dirname(datasets.__file__)) + '/'
        dataset_fp = datasets_fp + dataset_name
        experiment_name_list = split('_',experiment_name)
        experiment_name_list[-1] = 'train.csv'
        train_csv_fp = dataset_fp+'_'+'_'.join(experiment_name_list)
        experiment_name_list[-2] = '0'
        experiment_name_list[-1] = 'gt.csv'
        gt_csv_fp = dataset_fp+'_'+'_'.join(experiment_name_list)
        try:  # reads CSV
            str_triples = pd.read_csv(gt_csv_fp)
        except IOError as e:
            tp('f','Could not load ' + str(gt_csv_fp))
            raise IOError
        # loads into counts datastructure
        num_rows = str_triples.shape[0]
        for idx in xrange(num_rows):
            self.counts[self.r2i[str_triples.iloc[idx,1]],
                        self.e2i[str_triples.iloc[idx,0]],
                        self.e2i[str_triples.iloc[idx,2]]] += 1
        if self.exclude:
            try:  # reads exclude CSV
                str_triples = pd.read_csv(train_csv_fp)
            except IOError as e:
                tp('f','Could not load ' + str(train_csv_fp))
                raise IOError
            # excludes triples from ground truth rankings
            num_rows = str_triples.shape[0]
            for idx in xrange(num_rows):
                self.counts[self.r2i[str_triples.iloc[idx,1]],
                            self.e2i[str_triples.iloc[idx,0]],
                            self.e2i[str_triples.iloc[idx,2]]] = 0

    def score(self, s, r, o):
        if s is None:
            outputs = np.transpose([self.counts[r,:,o]])
            ids = np.transpose([range(self.counts.shape[1])])
            outputs = np.append(ids,outputs,1)
        elif o is None:
            outputs = np.transpose([self.counts[r,s,:]])
            ids = np.transpose([range(self.counts.shape[1])])
            outputs = np.append(ids,outputs,1)
        elif r is None:
            outputs = np.transpose([self.counts[:,s,o]])
            ids = np.transpose([range(self.counts.shape[0])])
            outputs = np.append(ids,outputs,1)
        return outputs

    def get_ranks(self,qtype,triple):
        subj,rel,obj = triple
        if qtype == 0:
            outputs = self.score(None,rel,obj)
            scores = outputs[:,1]
            idxs = np.argsort(scores)
            outputs = outputs[np.flip(idxs,0)]
            rank = np.where(outputs[:,0]==subj)
        elif qtype == 1:
            outputs = self.score(subj,None,obj)
            scores = outputs[:,1]
            idxs = np.argsort(scores)
            outputs = outputs[np.flip(idxs,0)]
            rank = np.where(outputs[:,0]==rel)
        elif qtype == 2:
            outputs = self.score(subj,rel,None)
            scores = outputs[:,1]
            idxs = np.argsort(scores)
            outputs = outputs[np.flip(idxs,0)]
            rank = np.where(outputs[:,0]==obj)
        return rank[0].tolist()[0]


if __name__ == "__main__":
    # creates train triples dataset
    dataset_name = 'demo'
    experiment_name = 'ex_0'
    dataset = TrainDataset(dataset_name,experiment_name+'_train',1,'random')
    # loads triples for training
    batch_size = 2
    num_threads = 1
    dataset_loader = DataLoader(dataset,batch_size=batch_size,shuffle=False,
                                num_workers=num_threads)
    # loop through data printing batch number and samples
    for idx_b, batch in enumerate(dataset_loader):
        print idx_b
        print batch

    # creates valid triples dataset
    dataset = PredictDataset(dataset_name,experiment_name+'_valid')
    # loads triples for training
    batch_size = 2
    num_threads = 1
    dataset_loader = DataLoader(dataset,batch_size=batch_size,shuffle=False,
                                num_workers=num_threads)
    # loop through data printing batch number and samples
    for idx_b, batch in enumerate(dataset_loader):
        print idx_b
        print batch