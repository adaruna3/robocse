import torch.nn as nn
import torch
from torch.autograd import Variable
import numpy as np

from data_utils import TriplesCounter

import pdb


class Analogy(nn.Module):
    def __init__(self,num_ents,num_rels,hidden_size,device,lmbda=0.0):
        """
        Creates an analogy model object
        :param num_ents: total number of entities
        :param num_rels: total number of relations
        :param hidden_size: dimensionality of the embedding
        :param lmbda: loss regularization for learning
        """
        super(Analogy, self).__init__()
        self.num_ents = num_ents
        self.num_rels = num_rels
        self.hidden_size = hidden_size
        self.lmbda = lmbda
        self.device = device
        self.ent_im_embeddings = \
            nn.Embedding(num_ents,hidden_size/4).to(self.device)
        self.ent_re_embeddings = \
            nn.Embedding(num_ents,hidden_size/4).to(self.device)
        self.rel_re_embeddings = \
            nn.Embedding(num_rels,hidden_size/4).to(self.device)
        self.rel_im_embeddings = \
            nn.Embedding(num_rels,hidden_size/4).to(self.device)
        self.ent_embeddings = \
            nn.Embedding(num_ents,hidden_size/2).to(self.device)
        self.rel_embeddings = \
            nn.Embedding(num_rels,hidden_size/2).to(self.device)
        self.softplus = nn.Softplus().to(self.device)
        self.init_embeddings()

    def init_embeddings(self):
        """
        Initializes the embedding weights
        """
        nn.init.xavier_uniform_(self.ent_re_embeddings.weight.data)
        nn.init.xavier_uniform_(self.ent_im_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_re_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_im_embeddings.weight.data)
        nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_embeddings.weight.data)

    def _calc(self,e_re_s,e_im_s,e_s,e_re_o,e_im_o,e_o,r_re,r_im,r):
        """
        Calculates the analogy score for a knowledge triple
        :param e_re_s: real part of subject (complex)
        :param e_im_s: imaginary part of subject (complex)
        :param e_s: real part of subject (distmult)
        :param e_re_o: real part of object (complex)
        :param e_im_o: imaginary part of object (complex)
        :param e_o: real part of object (distmult)
        :param r_re: real part of relation (complex)
        :param r_im: imaginary part of relation (complex)
        :param r: real part of relation (distmult)
        :return: analogy score of triple
        """
        return torch.sum(r_re * e_re_s * e_re_o +
                         r_re * e_im_s * e_im_o +
                         r_im * e_re_s * e_im_o -
                         r_im * e_im_s * e_re_o,
                         1,False) + \
               torch.sum(e_s*e_o*r,1,False)

    def forward(self,batch):
        """
        Feeds a batch of triples forward through the analogy model for training
        :param batch: input batch of triples
        :param labels: labels for input batch
        :return: returns average loss over the batch
        """
        batch_s = batch[:,:,0].reshape(batch.shape[0]*batch.shape[1])
        batch_r = batch[:,:,1].reshape(batch.shape[0]*batch.shape[1])
        batch_o = batch[:,:,2].reshape(batch.shape[0]*batch.shape[1])
        batch_y = batch[:,:,3].reshape(batch.shape[0]*batch.shape[1])
        # get the corresponding embeddings for calculations
        e_re_s = self.ent_re_embeddings(batch_s)
        e_im_s = self.ent_im_embeddings(batch_s)
        e_s = self.ent_embeddings(batch_s)
        r_re = self.rel_re_embeddings(batch_r)
        r_im = self.rel_im_embeddings(batch_r)
        r = self.rel_embeddings(batch_r)
        e_re_o = self.ent_re_embeddings(batch_o)
        e_im_o = self.ent_im_embeddings(batch_o)
        e_o = self.ent_embeddings(batch_o)
        # calculates the loss
        res = self._calc(e_re_s,e_im_s,e_s,e_re_o,e_im_o,e_o,r_re,r_im,r)
        emb_loss = torch.mean(self.softplus(- batch_y.float() * res))
        # included regularization term
        regul = torch.mean(e_re_s**2)+torch.mean(e_im_s**2)*torch.mean(e_s**2)+\
               torch.mean(e_re_o**2)+torch.mean(e_im_o**2)+torch.mean(e_o**2)+\
               torch.mean(r_re**2)+torch.mean(r_im**2)+torch.mean(r**2)
        # calculates loss to get what the framework will optimize
        loss = emb_loss + self.lmbda*regul
        return loss

    def predict(self,batch):
        """
        Feeds a batch of triples forward through the analogy model for inference
        :param batch:
        :return:
        """
        batch_s = batch[:,0]
        batch_r = batch[:,1]
        batch_o = batch[:,2]
        # get the corresponding embeddings for calculations
        p_re_s = self.ent_re_embeddings(Variable(torch.from_numpy(batch_s)))
        p_re_o = self.ent_re_embeddings(Variable(torch.from_numpy(batch_o)))
        p_re_r = self.rel_re_embeddings(Variable(torch.from_numpy(batch_r)))
        p_im_s = self.ent_im_embeddings(Variable(torch.from_numpy(batch_s)))
        p_im_o = self.ent_im_embeddings(Variable(torch.from_numpy(batch_o)))
        p_im_r = self.rel_im_embeddings(Variable(torch.from_numpy(batch_r)))
        p_s = self.ent_embeddings(Variable(torch.from_numpy(batch_s)))
        p_o = self.ent_embeddings(Variable(torch.from_numpy(batch_o)))
        p_r = self.rel_embeddings(Variable(torch.from_numpy(batch_r)))
        # calculates the score
        score = -self._calc(p_re_s, p_im_s, p_s,
                            p_re_o, p_im_o, p_o,
                            p_re_r, p_im_r, p_r)
        return score.cpu()

    def get_ranks(self,batch):
        batch_s = batch[:,:,0].reshape(batch.shape[0]*batch.shape[1])
        batch_r = batch[:,:,1].reshape(batch.shape[0]*batch.shape[1])
        batch_o = batch[:,:,2].reshape(batch.shape[0]*batch.shape[1])
        batch_q = batch[:,:,4].reshape(batch.shape[0]*batch.shape[1])
        batch_y = []
        for q_idx in xrange(len(batch_q)):
            # sets up the query
            if batch_q[q_idx] == 0:
                subj = Variable(torch.from_numpy(
                    np.arange(self.num_ents))).to(self.device)
                rel = batch_r[q_idx]
                obj = batch_o[q_idx]
            elif batch_q[q_idx] == 1:
                subj = batch_s[q_idx]
                rel = Variable(torch.from_numpy(
                    np.arange(self.num_rels))).to(self.device)
                obj = batch_o[q_idx]
            elif batch_q[q_idx] == 2:
                subj = batch_s[q_idx]
                rel = batch_r[q_idx]
                obj = Variable(torch.from_numpy(
                    np.arange(self.num_ents))).to(self.device)
            # gets embeddings
            e_re_s = self.ent_re_embeddings(subj)
            e_im_s = self.ent_im_embeddings(subj)
            e_s = self.ent_embeddings(subj)
            r_re = self.rel_re_embeddings(rel)
            r_im = self.rel_im_embeddings(rel)
            r = self.rel_embeddings(rel)
            e_re_o = self.ent_re_embeddings(obj)
            e_im_o = self.ent_im_embeddings(obj)
            e_o = self.ent_embeddings(obj)
            # calculates the rank
            scores = self._calc(e_re_s,e_im_s,e_s,e_re_o,e_im_o,e_o,r_re,r_im,r)
            ranks = np.flip(np.argsort(scores.cpu().detach().numpy(),0),0)
            # stores the rank
            pdb.set_trace()
            if batch_q[q_idx] == 0:
                batch_y.append(np.where(ranks==batch_s[q_idx])[0][0])
            elif batch_q[q_idx] == 1:
                batch_y.append(np.where(ranks==batch_r[q_idx])[0][0])
            elif batch_q[q_idx] == 2:
                batch_y.append(np.where(ranks==batch_o[q_idx])[0][0])
        return np.asarray(batch_y)


class AnalogyReduced(Analogy):
    """
    Reduced domain Analogy model
    """
    def __init__(self,
                 num_ents,
                 num_rels,
                 hidden_size,
                 device,
                 dataset_name,
                 experiment_name,
                 lmbda=0.0):
        super(AnalogyReduced, self).__init__(num_ents,
                                              num_rels,
                                              hidden_size,
                                              device,
                                              lmbda)
        self.tr_triples = TriplesCounter(dataset_name,experiment_name)

    def get_ranks(self,batch):
        batch_s = batch[:,:,0].reshape(batch.shape[0]*batch.shape[1])
        batch_r = batch[:,:,1].reshape(batch.shape[0]*batch.shape[1])
        batch_o = batch[:,:,2].reshape(batch.shape[0]*batch.shape[1])
        batch_q = batch[:,:,4].reshape(batch.shape[0]*batch.shape[1])
        batch_y = []
        for q_idx in xrange(len(batch_q)):
            # sets up the query
            if batch_q[q_idx] == 0:
                rel = batch_r[q_idx]
                obj = batch_o[q_idx]
                non_train = np.where(self.tr_triples.counts[rel,:,obj]==0)[0]
                subj = Variable(torch.from_numpy(non_train)).to(self.device)
            elif batch_q[q_idx] == 1:
                subj = batch_s[q_idx]
                obj = batch_o[q_idx]
                non_train = np.where(self.tr_triples.counts[:,subj,obj]==0)[0]
                rel = Variable(torch.from_numpy(non_train)).to(self.device)
            elif batch_q[q_idx] == 2:
                subj = batch_s[q_idx]
                rel = batch_r[q_idx]
                non_train = np.where(self.tr_triples.counts[rel,subj,:]==0)[0]
                obj = Variable(torch.from_numpy(non_train)).to(self.device)
            # gets embeddings
            e_re_s = self.ent_re_embeddings(subj)
            e_im_s = self.ent_im_embeddings(subj)
            e_s = self.ent_embeddings(subj)
            r_re = self.rel_re_embeddings(rel)
            r_im = self.rel_im_embeddings(rel)
            r = self.rel_embeddings(rel)
            e_re_o = self.ent_re_embeddings(obj)
            e_im_o = self.ent_im_embeddings(obj)
            e_o = self.ent_embeddings(obj)
            # calculates the rank
            scores = self._calc(e_re_s,e_im_s,e_s,e_re_o,e_im_o,e_o,r_re,r_im,r)
            ranks = np.flip(np.argsort(scores.cpu().detach().numpy(),0),0)
            # stores the rank
            if batch_q[q_idx] == 0:
                non_train_idx = np.where(non_train==batch_s[q_idx])[0][0]
                batch_y.append(np.where(ranks==non_train_idx)[0][0])
            elif batch_q[q_idx] == 1:
                non_train_idx = np.where(non_train==batch_r[q_idx])[0][0]
                batch_y.append(np.where(ranks==non_train_idx)[0][0])
            elif batch_q[q_idx] == 2:
                non_train_idx = np.where(non_train==batch_o[q_idx])[0][0]
                batch_y.append(np.where(ranks==non_train_idx)[0][0])
        return np.asarray(batch_y)



if __name__ == "__main__":
    from data_utils import TrainDataset,PredictDataset
    from torch.utils.data import DataLoader

    # creates triples train dataset
    dataset_name = 'demo'
    experiment_name = 'ex_0'
    dataset = TrainDataset(dataset_name,experiment_name+'_train',1,'random')

    # creates analogy model
    model = AnalogyReduced(len(dataset.e2i),
                           len(dataset.r2i),
                           100,
                           torch.device('cpu'),
                           dataset_name,
                           experiment_name+'_train')
    # creates reduced analogy model
    # model = Analogy(len(dataset.e2i),len(dataset.r2i),100,torch.device('cpu'))

    # batch loads triples for training
    batch_size = 2
    num_threads = 1
    dataset_loader = DataLoader(dataset,batch_size=batch_size,shuffle=False,
                                num_workers=num_threads)

    # tests forward prop with batches
    model.train()
    for idx_b, batch in enumerate(dataset_loader):

        loss = model.forward(batch)
        print "Loss of batch " + str(idx_b) + " is " + \
              str(loss.detach().numpy())

    # creates triples valid dataset
    dataset = PredictDataset(dataset_name,experiment_name+'_valid')
    dataset_loader = DataLoader(dataset,batch_size=batch_size,shuffle=False,
                                num_workers=num_threads)
    # tests ranking with batches
    model.eval()
    for idx_b, batch in enumerate(dataset_loader):
        ranks = model.get_ranks(batch)
        print 'Sro,sRo,srO rank for cabinet hasAff close: ' + str(ranks)
