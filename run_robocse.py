# torch imports
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# RoboCSE imports
from data_utils import TrainDataset
from models import Analogy
from trvate_utils import Evaluator,Trainer
from robocse_logging.viz_utils import tp,RoboCSETrainViz
import trained_models

# system imports
import pdb
from os.path import abspath,dirname
import argparse
from sys import stdin
from select import select
from re import split


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
    parser.add_argument('-e', dest='valid_freq', type=int, default=10,
                        nargs='?', help='Evaluation frequency')
    parser.add_argument('-t', dest='train', type=int, default=1,
                        nargs='?', help='Train=1,Test=0')
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

def training_validation_setup(cmd_args):
    # sets up triples training dataset
    dataset = TrainDataset(cmd_args.ds_name,cmd_args.exp_name,1,'random')
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
    # sets up for validation
    tr_validator = Evaluator(cmd_args.ds_name,
                             cmd_args.exp_name,
                             cmd_args.batch_size,
                             cmd_args.shuffle,
                             cmd_args.num_workers)
    return tr_trainer,tr_validator


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

    if args.train:
        # sets up for training/validation
        trainer,validator = training_validation_setup(args)
        # sets up for visualizing training/validation
        viz = RoboCSETrainViz(validator.dataset.i2r.values())

        # training and validation loop
        for epoch in xrange(args.num_epochs):
            # train
            total_loss = trainer.train_epoch()
            tp('d','Total loss is ' + str(total_loss) + ' for epoch ' + str(epoch))

            # validate
            if epoch % args.valid_freq == 0:
                performance = validator.evaluate(trainer.model)
                tp('i',"Current performance is: \n"+str(performance))
                viz.update(performance,total_loss)

        # save the trained model
        save_model(args.ds_name,args.exp_name,trainer.model.state_dict())