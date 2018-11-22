# torch imports
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# RoboCSE imports
from data_utils import KnowledgeTriplesDataset
from models import Analogy
from logging.viz_utils import tp
import trained_models
from os.path import abspath,dirname
import argparse
import pdb
from sys import stdin
from select import select


def parse_command_line():
    parser = argparse.ArgumentParser(description='RoboCSE Trainer')
    parser.add_argument('ds_name', type=str,
                        help='DataSet name')
    parser.add_argument('exp_name', type=str,
                        help='EXPeriment name for train,valid, & test')
    parser.add_argument('-b', dest='batch_size', type=int, default=50,
                        nargs='?', help='Batch size')
    parser.add_argument('-n', dest='num_workers', type=int, default=8,
                        nargs='?', help='Number of training threads')
    parser.add_argument('-s', dest='shuffle', type=int, default=0,
                        nargs='?', help='Shuffle bathces flag')
    parser.add_argument('-d', dest='d_size', type=int, default=100,
                        nargs='?', help='embedding Dimensionality')
    parser.add_argument('-o', dest='num_epochs', type=int, default=500,
                        nargs='?', help='number of epOchs to train')
    parser.add_argument('-m', dest='opt_method', type=str, default='sgd',
                        nargs='?', help='optimization Method to use')
    parser.add_argument('-p', dest='opt_params', type=float, default=1e-4,
                        nargs='?', help='optimization Parameters')
    parsed_args = parser.parse_args()
    tp('i','The current training parameters are: \n'+str(parsed_args))
    if not confirm_params():
        exit()
    return parsed_args


def confirm_params():
    tp('d','Continue training? (Y/n) waiting 10s ...')
    i, o, e = select( [stdin], [], [], 10 )
    if i:  # read input
        cont = stdin.readline().strip()
        if cont == 'Y' or cont == 'y':
            return True
        else:
            return False
    else:  # no input, start training
        return True


def initialize_optimizer(method,params,model):
    if method == "adagrad":
        try:
            lr,lr_decay,weight_decay = params
        except ValueError as e:
            tp('w','Parameters for adagrad are "-p lr lr_decay weight_decay"')
            raise argparse.ArgumentError
        optimizer = optim.Adagrad(model.parameters(),
                                  lr=lr,
                                  lr_decay=lr_decay,
                                  weight_decay=weight_decay)
    elif method == "adadelta":
        try:
            lr = params
        except ValueError as e:
            tp('f','Parameters for adadelta are "-p lr"')
            raise argparse.ArgumentError
        optimizer = optim.Adadelta(model.parameters(), lr=lr)
    elif method == "adam":
        try:
            lr,lr_decay,weight_decay = params
        except ValueError as e:
            tp('f','Parameters for adam are "-p lr"')
            raise argparse.ArgumentError
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif method == "sgd":
        try:
            lr= params
        except ValueError as e:
            tp('f','Parameters for sgd are "-p lr"')
            raise argparse.ArgumentError
        optimizer = optim.SGD(model.parameters(), lr=lr)
    else:
        tp('f','Optimization options are "adagrad","adadelta","adam","sgd"')
        raise argparse.ArgumentError
    return optimizer


def save_model(dataset_name,experiment_name,params):
    models_fp = abspath(dirname(trained_models.__file__)) + '/'
    model_fp = models_fp + dataset_name + '_' + experiment_name + '.pt'
    torch.save(params,model_fp)
    tp('s','model for experiment ' + experiment_name + ' on ' + dataset_name + \
       'trained and saved successfully.')
    tp('s','Written to: '+model_fp)

if __name__ == "__main__":
    # parses command line arguments
    args = parse_command_line()
    # sets up triples dataset
    dataset = KnowledgeTriplesDataset(args.ds_name,args.exp_name,1,'random')
    # sets up batch data loader
    dataset_loader = DataLoader(dataset,
                                batch_size=args.batch_size,
                                shuffle=args.shuffle,
                                num_workers=args.num_workers)
    # sets up model
    rcse = Analogy(len(dataset.e2i),len(dataset.r2i),args.d_size)
    # sets up optimization method
    tr_optimizer = initialize_optimizer(args.opt_method,args.opt_params,rcse)

    # set up for evaluating training

    # set up for visualizing evaluation

    # runs training
    for epoch in xrange(args.num_epochs):
        total_loss = 0.0
        for idx_b, batch in enumerate(dataset_loader):
            tr_optimizer.zero_grad()
            loss = rcse.forward(batch)
            total_loss += loss.detach().numpy()
            loss.backward()
            tr_optimizer.step()
        tp('d','Total loss is ' + str(total_loss) + ' for epoch ' + str(epoch))
    # save the trained model
    save_model(args.ds_name,args.exp_name,rcse.state_dict())