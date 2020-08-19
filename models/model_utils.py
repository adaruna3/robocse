from os.path import abspath, dirname
import numpy as np
from copy import copy, deepcopy
# torch imports
import torch
from torch.utils.data import DataLoader
from torch import tensor, from_numpy, no_grad, save, load, arange
from torch.autograd import Variable
import torch.optim as optim

# user module imports
from logger.terminal_utils import logout
import datasets.data_utils as data_utils
import models.standard_models as std_models

import pdb
import time

#######################################################
#  Standard Processors (finetune/offline)
#######################################################
class TrainBatchProcessor:
    def __init__(self, cmd_args):
        self.args = copy(cmd_args)
        self.dataset = data_utils.TripleDataset(self.args.dataset, self.args.neg_ratio)
        self.dataset.load_triple_set(self.args.set_name)
        self.dataset.load_known_ent_set()
        self.dataset.load_known_rel_set()
        self.dataset.load_current_ents_rels()
        self.data_loader = DataLoader(self.dataset,
                                      shuffle=True,
                                      batch_size=self.args.batch_size,
                                      num_workers=self.args.num_workers,
                                      collate_fn=collate_batch,
                                      pin_memory=True)

    def reset_data_loader(self):
        self.data_loader = DataLoader(self.dataset,
                                      shuffle=True,
                                      batch_size=self.args.batch_size,
                                      num_workers=self.args.num_workers,
                                      collate_fn=collate_batch,
                                      pin_memory=True)

    def process_epoch(self, model, optimizer):
        model_was_training = model.training
        if not model_was_training:
            model.train()

        total_loss = 0.0
        for idx_b, batch in enumerate(self.data_loader):
            bh, br, bt, by = batch
            optimizer.zero_grad()
            batch_loss = model.forward(bh.contiguous().to(self.args.device),
                                       br.contiguous().to(self.args.device),
                                       bt.contiguous().to(self.args.device),
                                       by.contiguous().to(self.args.device))
            batch_loss.backward()
            optimizer.step()
            total_loss += batch_loss.item()
        return total_loss


class DevBatchProcessor:
    def __init__(self, cmd_args):
        self.args = copy(cmd_args)
        self.dataset = data_utils.TripleDataset(self.args.dataset, self.args.neg_ratio)
        self.dataset.load_triple_set(self.args.set_name)
        self.dataset.load_mask(cmd_args.dataset_fps)
        self.dataset.load_known_ent_set()
        self.dataset.load_known_rel_set()
        self.batch_size = 10
        self.data_loader = DataLoader(self.dataset,
                                      shuffle=False,
                                      batch_size=self.batch_size,
                                      num_workers=self.args.num_workers,
                                      collate_fn=collate_batch,
                                      pin_memory=True)
        self.cutoff = int(self.args.valid_cutoff / self.batch_size) if self.args.valid_cutoff is not None else None

    def process_epoch(self, model):
        model_was_training = model.training
        if model_was_training:
            model.eval()

        h_ranks = np.ndarray(shape=0, dtype=np.float64)
        t_ranks = np.ndarray(shape=0, dtype=np.float64)
        with no_grad():
            for idx_b, batch in enumerate(self.data_loader):
                if self.cutoff is not None:  # validate on less triples for large datasets
                    if idx_b > self.cutoff:
                        break

                if self.args.cuda and torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # get ranks for each triple in the batch
                bh, br, bt, by = batch
                h_ranks = np.append(h_ranks, self._rank_head(model, bh, br, bt), axis=0)
                t_ranks = np.append(t_ranks, self._rank_tail(model, bh, br, bt), axis=0)

        # calculate hits & mrr
        hits10_h = np.count_nonzero(h_ranks <= 10) / len(h_ranks)
        hits10_t = np.count_nonzero(t_ranks <= 10) / len(t_ranks)
        hits10 = (hits10_h + hits10_t) / 2.0
        mrr = np.mean(np.concatenate((1 / h_ranks, 1 / t_ranks), axis=0))

        return hits10, mrr

    def _rank_head(self, model, h, r, t):
        rank_heads = Variable(from_numpy(np.arange(len(self.dataset.e2i)))).repeat(h.shape[0], 1)
        scores = model.predict(rank_heads.contiguous().to(self.args.device),
                               r.unsqueeze(-1).contiguous().to(self.args.device),
                               t.unsqueeze(-1).contiguous().to(self.args.device))
        ranks = []
        known_ents = np.asarray(self.dataset.known_ents, dtype=np.int64)
        for i in range(scores.shape[0]):
            scores_ = copy(scores[i, :])
            scores_ = np.stack((scores_, np.arange(len(self.dataset.e2i))), axis=-1)
            if (int(r[i].numpy()), int(t[i].numpy())) in self.dataset.h_mask:
                h_mask = copy(self.dataset.h_mask[(int(r[i].numpy()), int(t[i].numpy()))])
                h_mask.remove(int(h[i].numpy()))
                ents = known_ents[np.isin(known_ents, h_mask, True, True)]
            else:
                ents = known_ents
            filtered_scores = scores_[np.isin(scores_[:, -1], ents, True), :]
            filtered_ent_idx = int(np.where(filtered_scores[:, -1] == int(h[i].numpy()))[0])
            ranks_ = np.argsort(filtered_scores[:, 0], 0)
            ranks.append(int(np.where(ranks_ == filtered_ent_idx)[0])+1)
        return ranks

    def _rank_tail(self, model, h, r, t):
        rank_tails = Variable(from_numpy(np.arange(len(self.dataset.e2i)))).repeat(t.shape[0], 1)
        scores = model.predict(h.unsqueeze(-1).contiguous().to(self.args.device),
                               r.unsqueeze(-1).contiguous().to(self.args.device),
                               rank_tails.contiguous().to(self.args.device))
        ranks = []
        known_ents = np.asarray(self.dataset.known_ents, dtype=np.int64)
        for i in range(scores.shape[0]):
            scores_ = copy(scores[i, :])
            scores_ = np.stack((scores_, np.arange(len(self.dataset.e2i))), axis=-1)
            if (int(h[i].numpy()), int(r[i].numpy())) in self.dataset.t_mask:
                t_mask = copy(self.dataset.t_mask[(int(h[i].numpy()), int(r[i].numpy()))])
                t_mask.remove(int(t[i].numpy()))
                ents = known_ents[np.isin(known_ents, t_mask, True, True)]
            else:
                ents = known_ents
            filtered_scores = scores_[np.isin(scores_[:, -1], ents, True), :]
            filtered_ent_idx = int(np.where(filtered_scores[:, -1] == int(t[i].numpy()))[0])
            ranks_ = np.argsort(filtered_scores[:, 0], 0)
            ranks.append(int(np.where(ranks_ == filtered_ent_idx)[0])+1)
        return ranks


def collate_batch(batch):
    batch = tensor(batch)
    batch_h = batch[:, :, 0].flatten()
    batch_r = batch[:, :, 1].flatten()
    batch_t = batch[:, :, 2].flatten()
    batch_y = batch[:, :, 3].flatten()
    return batch_h, batch_r, batch_t, batch_y


def init_model(args):
    model = None
    if args.model == "transe":
        model = std_models.TransE(args.num_ents, args.num_rels, args.hidden_size, args.margin,
                                  args.neg_ratio, args.batch_size, args.device)
        model.to(args.device, non_blocking=True)
    elif args.model == "analogy":
        model = std_models.Analogy(args.num_ents, args.num_rels, args.hidden_size, args.device)
        model.to(args.device, non_blocking=True)
    else:
        logout("The model '" + str(args.model) + "' to be used is not implemented.", "f")
        exit()
    return model


def init_optimizer(args, model):
    optim_model = model
    optimizer = None
    if args.opt_method == "adagrad":
        try:
            lr = args.opt_params[0]
            optimizer = optim.Adagrad(optim_model.parameters(), lr=lr)
        except ValueError as e:
            logout("Parameters for adagrad are [-op lr]", "f")
            exit()
    elif args.opt_method == "adadelta":
        try:
            lr = args.opt_params[0]
            optimizer = optim.Adadelta(optim_model.parameters(), lr=lr)
        except ValueError as e:
            logout("Parameters for adadelta are [-op lr]", "f")
            exit()
    elif args.opt_method == "adam":
        try:
            lr = args.opt_params[0]
            optimizer = optim.Adam(optim_model.parameters(), lr=lr)
        except ValueError as e:
            logout("Parameters for adam are [-op lr]", "f")
            exit()
    elif args.opt_method == "sgd":
        try:
            lr = args.opt_params[0]
            optimizer = optim.SGD(optim_model.parameters(), lr=lr)
        except ValueError as e:
            logout("Parameters for sgd are [-op lr]", "f")
            exit()
    else:
        logout("Optimization options are 'adagrad','adadelta','adam','sgd'", "f")
        exit()

    return optimizer


def save_model(args, model):
    checkpoints_fp = abspath(dirname(__file__)) + "/checkpoints/"
    checkpoint_name = str(args.tag) + "__"
    checkpoint_name += str(args.dataset) + "_"
    checkpoint_name += "mt" + str(args.model) + "_"
    checkpoint_name += "ln" + str(args.log_num)

    save_checkpoint(model.state_dict(), checkpoints_fp + checkpoint_name)


def save_checkpoint(params, filename):
    try:
        torch.save(params, filename)
        # logout('Written to: ' + filename)
    except Exception as e:
        logout("Could not save: " + filename, "w")
        raise e


def load_model(args, model):
    checkpoints_fp = abspath(dirname(__file__)) + "/checkpoints/"
    checkpoint_name = str(args.tag) + "__"
    checkpoint_name += str(args.dataset) + "_"
    checkpoint_name += "mt" + str(args.model) + "_"
    checkpoint_name += "ln" + str(args.log_num)

    model = load_checkpoint(model, checkpoints_fp + checkpoint_name)
    return model


def load_checkpoint(model, filename):
    try:
        model.load_state_dict(load(filename), strict=False)
    except Exception as e:
        logout("Could not load: " + filename, "w")
        raise e
    return model


def evaluate_model(args, sess, batch_processors, model):
    performances = np.ndarray(shape=(0, 2))
    for valid_sess in range(args.num_sess):
        eval_bp = batch_processors[valid_sess]
        performance = eval_bp.process_epoch(model)
        performances = np.append(performances, [performance], axis=0)
    return performances


class EarlyStopTracker:
    def __init__(self, args):
        self.args = args
        self.num_epoch = args.num_epochs
        self.epoch = 0
        self.valid_freq = args.valid_freq
        self.patience = args.patience
        self.early_stop_trigger = -int(self.patience / self.valid_freq)
        self.last_early_stop_value = 0.0
        self.best_performances = None
        self.best_measure = 0.0
        self.best_epoch = None

    def continue_training(self):
        return not bool(self.epoch > self.num_epoch or self.early_stop_trigger > 0)

    def get_epoch(self):
        return self.epoch

    def validate(self):
        return bool(self.epoch % self.valid_freq == 0)

    def update_best(self, sess, performances, model):
        measure = performances[sess, 1]
        # checks for new best model and saves if so
        if measure > self.best_measure:
            self.best_measure = copy(measure)
            self.best_epoch = copy(self.epoch)
            self.best_performances = np.copy(performances)
            save_model(self.args, model)
        # checks for reset of early stop trigger
        if measure - 0.01 > self.last_early_stop_value:
            self.last_early_stop_value = copy(measure)
            self.early_stop_trigger = -int(self.patience / self.valid_freq)
        else:
            self.early_stop_trigger += 1
        # adjusts valid frequency throughout training
        if self.epoch >= 400:
            self.early_stop_trigger = self.early_stop_trigger * self.valid_freq / 50.0
            self.valid_freq = 50
        elif self.epoch >= 200:
            self.early_stop_trigger = self.early_stop_trigger * self.valid_freq / 25.0
            self.valid_freq = 25
        elif self.epoch >= 50:
            self.early_stop_trigger = self.early_stop_trigger * self.valid_freq / 10.0
            self.valid_freq = 10

    def step_epoch(self):
        self.epoch += 1

    def get_best(self):
        return self.best_performances, self.best_epoch



if __name__ == "__main__":
    # TODO add unit tests below
    pass
