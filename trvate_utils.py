import trained_models
from os.path import abspath,dirname
import numpy as np
from logging.viz_utils import tp
from data_utils import PredictDataset
from torch.utils.data import DataLoader

from copy import copy

class Evaluator():
    def __init__(self,
                 dataset_name,
                 experiment_name,
                 batch_size,
                 shuffle,
                 num_workers):
        """
        :param dataset_name:
        :param experiment_name:
        """
        self.dataset = PredictDataset(dataset_name,experiment_name)
        self.dataset_loader = DataLoader(self.dataset,
                                         batch_size=batch_size,
                                         shuffle=shuffle,
                                         num_workers=num_workers)

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
        total = np.zeros((3,len(self.dataset.r2i),1),np.float64)
        metric = np.zeros((3,len(self.dataset.r2i),4),np.float64)

        # evaluate the model over the triples set
        for idx_b, batch in enumerate(self.dataset_loader):
                m1_o = batch[:,:,4].flatten()
                m2_o = model.get_ranks(batch)
                # gets AMMR and Hits@X* metrics
                amrr = self.get_amrr(m1_o.numpy(),m2_o)
                hits = abs(m1_o.numpy()-m2_o)
                # store results
                for idx_q in xrange(3):
                    for idx_r in xrange(len(self.dataset.r2i)):
                        idxs = np.logical_and((batch[:,:,1].flatten()==idx_r),
                                              (batch[:,:,3].flatten()==idx_q))
                        idxs = np.where(idxs)
                        metric[idx_q,idx_r,0] += \
                            np.sum(amrr[idxs[0]])  # AMRR
                        metric[idx_q,idx_r,1] += \
                            np.count_nonzero(hits[idxs[0]]<=10)  # Hits@10
                        metric[idx_q,idx_r,2] += \
                            np.count_nonzero(hits[idxs[0]]<=5)  # Hits@5
                        metric[idx_q,idx_r,3] += \
                            np.count_nonzero(hits[idxs[0]]<=1)  # Hits@1
                        total[idx_q,idx_r,0] += len(idxs[0])  # increases total
                # debug prints
                #self.debug_output(batch,hits,metric,total)

        # set model back to previous state
        if model_was_training:
            model.train()
        return metric/total

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


class Trainer():
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


if __name__ == "__main__":
    import torch
    from models import Analogy
    import pdb


    va_eval = Evaluator('demo','ex',2,False,1)
    va_model = Analogy(len(va_eval.dataset.e2i),len(va_eval.dataset.r2i),100)
    models_fp = abspath(dirname(trained_models.__file__)) + '/'
    model_fp = models_fp + 'demo' + '_' + 'train' + '.pt'
    va_model.load_state_dict(torch.load(model_fp))
    tp('s','Model evaluated! Metrics below: \n'+str(va_eval.evaluate(va_model)))
