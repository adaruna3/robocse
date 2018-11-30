import numpy as np
from argparse import ArgumentError

from torch.utils.data import DataLoader
import torch.optim as optim

from robocse_logging.viz_utils import tp
from data_utils import PredictDataset
from models import Analogy
from data_utils import TrainDataset

from copy import copy

import pdb


class Evaluator:
    def __init__(self,
                 dataset_name,
                 experiment_name,
                 batch_size,
                 shuffle,
                 num_workers,
                 batch_cutoff=None):
        """
        :param dataset_name:
        :param experiment_name:
        """
        self.dataset = PredictDataset(dataset_name,experiment_name)
        self.dataset_loader = DataLoader(self.dataset,
                                         batch_size=batch_size,
                                         shuffle=shuffle,
                                         num_workers=num_workers)
        self.cutoff = batch_cutoff

    def evaluate(self,model):
        """
        Runs validation on the current model reporting metrics
        :param model: model to validate
        :return: metrics
        """
        # set model for evaluation
        model_was_training = model.training
        if model_was_training:
            model.eval()

        # initialize return variables
        num_samples = np.zeros((3,len(self.dataset.r2i),1),np.float64)
        metric = np.zeros((3,len(self.dataset.r2i),4),np.float64)
        total_loss = 0.0

        # evaluate the model over the triples set
        for idx_b, batch in enumerate(self.dataset_loader):
            if (self.cutoff is not None) and (idx_b > self.cutoff):
                break  # break if not going through all batches
            # gets total loss
            total_loss += model.forward(batch).detach().numpy()
            # gets AMMR and Hits@X* metrics
            m1_o = batch[:,:,5].flatten()
            m2_o = model.get_ranks(batch)
            amrr = self.get_amrr(m1_o.numpy(),m2_o)
            hits = abs(m1_o.numpy()-m2_o)
            # store results
            for idx_q in xrange(3):
                for idx_r in xrange(len(self.dataset.r2i)):
                    idxs = np.logical_and((batch[:,:,1].flatten()==idx_r),
                                          (batch[:,:,4].flatten()==idx_q))
                    idxs = np.where(idxs)
                    metric[idx_q,idx_r,0] += \
                        np.sum(amrr[idxs[0]])  # AMRR
                    metric[idx_q,idx_r,1] += \
                        np.count_nonzero(hits[idxs[0]]<=10)  # Hits@10
                    metric[idx_q,idx_r,2] += \
                        np.count_nonzero(hits[idxs[0]]<=5)  # Hits@5
                    metric[idx_q,idx_r,3] += \
                        np.count_nonzero(hits[idxs[0]]<=1)  # Hits@1
                    num_samples[idx_q,idx_r,0] += len(idxs[0])

        # set model back to previous state
        if model_was_training:
            model.train()

        return metric/num_samples, total_loss

    def get_amrr(self,o1,o2):
        # adjust o1
        o1 = copy(o1)
        idxs = np.where(o1<o2)
        o1[idxs] -= 1
        # adjust o1 again
        idxs = np.where(o1>o2)
        o1[idxs] += 1
        # calculate mrr
        result = abs(o2-o1)**-1.0
        idxs = np.where(o1==o2)
        result[idxs] = 1.0
        return result

    def debug_output(self,batch,hits,metric,total):
        tp('d','Current batch is: \n' + str(batch))
        tp('d','Current relation total is: \n' + str(total))
        tp('d','Batch hits are: \n' + str(hits))
        tp('d','All hits are: \n' + str(metric[:,:,1:]))


class Trainer:
    def __init__(self,data_loader,optimizer,model):
        self.data_loader = data_loader
        self.optimizer = optimizer
        self.model = model

    def train_epoch(self):
        total_loss = 0.0
        for idx_b, batch in enumerate(self.data_loader):
            self.optimizer.zero_grad()
            loss = self.model.forward(batch)
            total_loss += loss.detach().numpy()
            loss.backward()
            self.optimizer.step()
        return total_loss


def validation_setup(cmd_args):
    # sets up for validation
    tr_evaluator = Evaluator(cmd_args.ds_name,
                             cmd_args.exp_name+'_train',
                             cmd_args.batch_size,
                             cmd_args.shuffle,
                             cmd_args.num_workers,
                             cmd_args.batch_cutoff)
    va_evaluator = Evaluator(cmd_args.ds_name,
                             cmd_args.exp_name+'_valid',
                             cmd_args.batch_size,
                             cmd_args.shuffle,
                             cmd_args.num_workers)
    return tr_evaluator,va_evaluator


def training_setup(cmd_args):
    # sets up triples training dataset
    dataset = TrainDataset(cmd_args.ds_name,
                           cmd_args.exp_name,
                           cmd_args.neg_ratio,
                           cmd_args.neg_method)
    # sets up batch data loader
    dataset_loader = DataLoader(dataset,
                                batch_size=cmd_args.batch_size,
                                shuffle=cmd_args.shuffle,
                                num_workers=cmd_args.num_workers)
    # sets up model
    model = Analogy(len(dataset.e2i),len(dataset.r2i),cmd_args.d_size)
    # sets up optimization method
    tr_optimizer = initialize_optimizer(cmd_args.opt_method,
                                        cmd_args.opt_params,model)
    # sets up for training
    tr_trainer = Trainer(dataset_loader,tr_optimizer,model)

    return tr_trainer


def initialize_optimizer(method,params,model):
    if method == "adagrad":
        try:
            lr,lr_decay,weight_decay = params
        except ValueError as e:
            tp('w','Parameters for adagrad are "-p (lr,lr_decay,weight_decay)"')
            raise ArgumentError
        optimizer = optim.Adagrad(model.parameters(),
                                  lr=lr,
                                  lr_decay=lr_decay,
                                  weight_decay=weight_decay)
    elif method == "adadelta":
        try:
            lr = params
        except ValueError as e:
            tp('f','Parameters for adadelta are "-p lr"')
            raise ArgumentError
        optimizer = optim.Adadelta(model.parameters(), lr=lr)
    elif method == "adam":
        try:
            lr,lr_decay,weight_decay = params
        except ValueError as e:
            tp('f','Parameters for adam are "-p lr"')
            raise ArgumentError
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif method == "sgd":
        try:
            lr= params
        except ValueError as e:
            tp('f','Parameters for sgd are "-p lr"')
            raise ArgumentError
        optimizer = optim.SGD(model.parameters(), lr=lr)
    else:
        tp('f','Optimization options are "adagrad","adadelta","adam","sgd"')
        raise ArgumentError
    return optimizer


if __name__ == "__main__":
    class example_args:
        def __init__(self):
            self.ds_name = 'demo'
            self.exp_name = 'ex'
            self.batch_size = 2
            self.num_workers = 1
            self.shuffle = 0
            self.d_size = 100
            self.num_epochs = 20
            self.opt_method = 'sgd'
            self.opt_params = 1e-1
            self.valid_freq = 1
            self.train = 1
            self.batch_cutoff = None
    temp_args = example_args()
    # check training
    trainer = training_setup(temp_args)
    tl = trainer.train_epoch()
    tp('s','Model trained!: ' + str(tl))
    # check evaluation
    tr_eval,va_eval = validation_setup(temp_args)
    tr_performance,tr_loss = tr_eval.evaluate(trainer.model)
    va_performance,va_loss = va_eval.evaluate(trainer.model)
    tp('s','Model evaluated! Trian: \n'+str(tr_performance))
    tp('s','Model evaluated! Valid: \n'+str(va_performance))
