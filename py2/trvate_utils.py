import numpy as np
from argparse import ArgumentError

from torch.utils.data import DataLoader
import torch.optim as optim
import torch

from robocse_logging.viz_utils import tp
from data_utils import PredictDataset
from models import Analogy,AnalogyReduced
from data_utils import TrainDataset
import trained_models

from copy import copy
from os.path import abspath,dirname

import pdb


class Evaluator:
    def __init__(self,
                 dataset_name,
                 experiment_name,
                 batch_size,
                 num_workers,
                 device,
                 exclude_train,
                 batch_cutoff=None):
        """
        :param dataset_name:
        :param experiment_name:
        """
        self.dataset = PredictDataset(dataset_name,
                                      experiment_name,
                                      exclude_train=exclude_train)
        self.dataset_loader = DataLoader(self.dataset,
                                         batch_size=batch_size,
                                         shuffle=False,
                                         num_workers=num_workers,
                                         collate_fn=valid_collate)
        self.cutoff = batch_cutoff
        self.device = device

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
        for idx_b,batch in enumerate(self.dataset_loader):
            bs,br,bo,by,bq,bk = batch
            if (self.cutoff is not None) and (idx_b > self.cutoff):
                break  # break if not going through all batches
            # gets total loss
            loss = copy(model.forward(bs.to(self.device),br.to(self.device),
                                      bo.to(self.device),by.to(self.device)).cpu())
            total_loss += loss.detach().numpy()

            # gets AMMR and Hits@X* metrics
            ranks1 = bk + 1.0
            ranks2 = model.get_ranks(bs.to(self.device),br.to(self.device),
                                     bo.to(self.device),bq.to(self.device))+1.0
            hits = np.abs(ranks1.numpy()-ranks2)+1.0
            amrr = 1.0/hits
            # store results
            for idx_q in xrange(3):
                if idx_q == 1:  # handles sRo queries
                    idxs = np.where(bq==idx_q)
                    metric[idx_q,:,0] += \
                        np.sum(amrr[idxs[0]])  # AMRR
                    metric[idx_q,:,1] += \
                        np.count_nonzero(hits[idxs[0]]<=10)  # Hits@10
                    metric[idx_q,:,2] += \
                        np.count_nonzero(hits[idxs[0]]<=5)  # Hits@5
                    metric[idx_q,:,3] += \
                        np.count_nonzero(hits[idxs[0]]<=1)  # Hits@1
                    num_samples[idx_q,:,0] += len(idxs[0])
                else:
                    for idx_r in xrange(len(self.dataset.r2i)):
                        idxs = np.logical_and((br==idx_r),(bq==idx_q))
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

    def debug_output(self,batch,hits,metric,total):
        tp('d','Current batch is: \n' + str(batch))
        tp('d','Current relation total is: \n' + str(total))
        tp('d','Batch hits are: \n' + str(hits))
        tp('d','All hits are: \n' + str(metric[:,:,1:]))


class Trainer:
    def __init__(self,data_loader,optimizer,model,device):
        self.data_loader = data_loader
        self.optimizer = optimizer
        self.model = model
        self.device = device

    def train_epoch(self):
        total_loss = 0.0
        for idx_b,batch in enumerate(self.data_loader):
            bs,br,bo,by = batch
            self.optimizer.zero_grad()
            loss = self.model.forward(
                bs.contiguous().cuda(non_blocking=True),
                br.contiguous().cuda(non_blocking=True),
                bo.contiguous().cuda(non_blocking=True),
                by.contiguous().cuda(non_blocking=True))
            total_loss += loss.cpu().detach().numpy()
            loss.backward()
            self.optimizer.step()
        return total_loss


def validation_setup(cmd_args):
    # sets up for validation
    va_evaluator = Evaluator(cmd_args.ds_name,
                             cmd_args.exp_name+'_valid',
                             cmd_args.batch_size,
                             cmd_args.num_workers,
                             cmd_args.device,
                             cmd_args.exclude_train)
    return va_evaluator


def training_setup(cmd_args):
    # sets up triples training dataset
    dataset = TrainDataset(cmd_args.ds_name,
                           cmd_args.exp_name+'_train',
                           cmd_args.neg_ratio,
                           cmd_args.neg_method)
    # sets up batch data loader
    dataset_loader = DataLoader(dataset,
                                batch_size=cmd_args.batch_size,
                                shuffle=cmd_args.shuffle,
                                num_workers=cmd_args.num_workers,
                                collate_fn=train_collate,
                                pin_memory=True)
    # sets up model
    if cmd_args.exclude_train:
        model = AnalogyReduced(len(dataset.e2i),
                               len(dataset.r2i),
                               cmd_args.d_size,
                               cmd_args.device,
                               cmd_args.ds_name,
                               cmd_args.exp_name+'_train')
    else:
        model = Analogy(len(dataset.e2i),
                        len(dataset.r2i),
                        cmd_args.d_size,
                        cmd_args.device)
    model.to(cmd_args.device,non_blocking=True)
    # sets up optimization method
    tr_optimizer = initialize_optimizer(cmd_args.opt_method,
                                        cmd_args.opt_params,model)
    # sets up for training
    tr_trainer = Trainer(dataset_loader,tr_optimizer,model,cmd_args.device)

    return tr_trainer


def train_collate(batch):
    batch = torch.tensor(batch)
    batch_s = batch[:,:,0].flatten()
    batch_r = batch[:,:,1].flatten()
    batch_o = batch[:,:,2].flatten()
    batch_y = batch[:,:,3].flatten()
    return batch_s,batch_r,batch_o,batch_y


def valid_collate(batch):
    batch = torch.tensor(batch)
    batch_s = batch[:,:,0].flatten()
    batch_r = batch[:,:,1].flatten()
    batch_o = batch[:,:,2].flatten()
    batch_y = batch[:,:,3].flatten()
    batch_q = batch[:,:,4].flatten()
    batch_k = batch[:,:,5].flatten()
    return batch_s,batch_r,batch_o,batch_y,batch_q,batch_k

def testing_setup(cmd_args,fold):
    te_tester = Evaluator(cmd_args.ds_name,
                          cmd_args.exp_name+'_'+str(fold)+'_test',
                          cmd_args.batch_size,
                          cmd_args.num_workers,
                          cmd_args.device,
                          cmd_args.exclude_train)
    # sets up model
    if cmd_args.exclude_train:
        te_model = AnalogyReduced(len(te_tester.dataset.e2i),
                                  len(te_tester.dataset.r2i),
                                  cmd_args.d_size,
                                  cmd_args.device,
                                  cmd_args.ds_name,
                                  cmd_args.exp_name+'_'+str(fold)+'_train')
        models_fp = abspath(dirname(trained_models.__file__)) + '/'
        model_fp = models_fp + cmd_args.ds_name + '_' + \
                   cmd_args.exp_name+'_'+str(fold) + '.pt'
        te_model.load_state_dict(torch.load(model_fp))
        te_model.eval()
    else:
        te_model = Analogy(len(te_tester.dataset.e2i),
                           len(te_tester.dataset.r2i),
                           cmd_args.d_size,
                           cmd_args.device)
        models_fp = abspath(dirname(trained_models.__file__)) + '/'
        model_fp = models_fp + cmd_args.ds_name + '_' + \
                   cmd_args.exp_name+'_'+str(fold) + '.pt'
        te_model.load_state_dict(torch.load(model_fp))
        te_model.eval()
    return te_tester, te_model


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
            lr = params[0]
        except ValueError as e:
            tp('f','Parameters for adadelta are "-p lr"')
            raise ArgumentError
        optimizer = optim.Adadelta(model.parameters(), lr=lr)
    elif method == "adam":
        try:
            lr = params[0]
        except ValueError as e:
            tp('f','Parameters for adam are "-p lr"')
            raise ArgumentError
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif method == "sgd":
        try:
            lr= params[0]
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
            self.ds_name = 'sd_thor'
            self.exp_name = 'tg_all_0'
            self.batch_size = 200
            self.num_workers = 1
            self.shuffle = 0
            self.d_size = 100
            self.num_epochs = 20
            self.opt_method = 'sgd'
            self.opt_params = [1e-1]
            self.valid_freq = 1
            self.train = 1
            self.batch_cutoff = None
            self.neg_ratio = 2
            self.neg_method = 'random'
            self.exclude_train = 1
            self.cuda = 1
            self.device = torch.device('cuda')
    temp_args = example_args()
    # check training
    trainer = training_setup(temp_args)
    tl = trainer.train_epoch()
    tp('s','Model trained!: ' + str(tl))
    # check evaluation
    va_eval = validation_setup(temp_args)
    va_performance,va_loss = va_eval.evaluate(trainer.model)
    tp('s','Model evaluated! Valid: \n'+str(va_performance))
