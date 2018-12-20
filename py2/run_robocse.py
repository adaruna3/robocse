# torch imports
import torch

# RoboCSE imports
from trvate_utils import training_setup,validation_setup,testing_setup
from robocse_logging.viz_utils import tp,valid_visualization_setup
import trained_models

# system imports
import pdb
from os.path import abspath,dirname
import argparse
from sys import stdin
from select import select
import numpy as np
from time import time


def parse_command_line():
    parser = argparse.ArgumentParser(description='RoboCSE Trainer')
    parser.add_argument('ds_name', type=str,
                        help='DataSet name')
    parser.add_argument('exp_name', type=str,
                        help='EXPeriment NAME for train,valid, & test')
    parser.add_argument('-bs', dest='batch_size', type=int, default=50,
                        nargs='?', help='Batch size')
    parser.add_argument('-n', dest='num_workers', type=int, default=8,
                        nargs='?', help='Number of training threads')
    parser.add_argument('-s', dest='shuffle', type=int, default=1,
                        nargs='?', help='Shuffle bathces flag')
    parser.add_argument('-d', dest='d_size', type=int, default=100,
                        nargs='?', help='embedding Dimensionality')
    parser.add_argument('-o', dest='num_epochs', type=int, default=500,
                        nargs='?', help='number of epOchs to train')
    parser.add_argument('-m', dest='opt_method', type=str, default='sgd',
                        nargs='?', help='optimization Method to use')
    parser.add_argument('-p', dest='opt_params', type=float, default=1e-4,
                        nargs='+', help='optimization Parameters')
    parser.add_argument('-e', dest='valid_freq', type=int, default=10,
                        nargs='?', help='Evaluation frequency')
    parser.add_argument('-t', dest='train', type=int, default=1,
                        nargs='?', help='Train=1,Test=0')
    parser.add_argument('-bc', dest='batch_cutoff', type=int, default=None,
                        nargs='?', help='Negative sampling Ratio')
    parser.add_argument('-nr', dest='neg_ratio', type=int, default=9,
                        nargs='?', help='Cutoff for training visualization')
    parser.add_argument('-nm', dest='neg_method', type=str, default='random',
                        nargs='?', help='Negative sampling Method')
    parser.add_argument('-c', dest='cuda', type=int, default=1,
                        nargs='?', help='Run on GPU?')
    parser.add_argument('-et', dest='exclude_train', type=int, default=1,
                        nargs='?', help='Exclude train triples from ranks?')
    parser.add_argument('-k', dest='num_folds', type=int, default=5,
                        nargs='?', help='Testing number of folds')
    parser.add_argument('-en', dest='exp_num', type=int, default=0,
                        nargs='?', help='Experiment number')

    parsed_args = parser.parse_args()
    if parsed_args.train:
        tp('i','The current training parameters are: \n'+str(parsed_args))
    else:
        tp('i','The current testing parameters are: \n'+str(parsed_args))

    if not confirm_params():
        exit()
    return parsed_args


def confirm_params():
    tp('d','Continue training? (Y/n) waiting 10s ...')
    i, o, e = select([stdin], [], [], 10.0)
    if i:  # read input
        cont = stdin.readline().strip()
        if cont == 'Y' or cont == 'y' or cont == '':
            return True
        else:
            return False
    else:  # no input, start training
        return True


def save_model(cmd_args,params,current,best):
    ds_name = cmd_args.ds_name
    exp_name = cmd_args.exp_name
    exp_num = cmd_args.exp_num
    mrr = np.mean(current[0,:,1]+current[2,:,1])
    if mrr > best:
        best = mrr
        models_fp = abspath(dirname(trained_models.__file__)) + '/'
        model_fp = models_fp+ds_name+'_'+exp_name+'_'+str(exp_num)+'.pt'
        #model_fp = models_fp+ds_name+'_'+exp_name+'.pt'
        torch.save(params,model_fp)
        tp('s','New best model for ' + exp_name + ' on ' + ds_name)
        tp('s','Written to: '+model_fp)
    return best

if __name__ == "__main__":
    # parses command line arguments
    args = parse_command_line()

    # select hardware to use
    if args.cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')

    start = time()

    if args.train:
        # sets up for training
        trainer = training_setup(args)
        # sets up for validation
        #tr_eval,va_eval = validation_setup(args)
        va_eval = validation_setup(args)
        # sets up for visualizing training/validation
        #tr_viz,va_viz = valid_visualization_setup(tr_eval,va_eval)
        #va_viz = valid_visualization_setup(va_eval)
        # sets up for saving trained models
        best_performance = 0.0
        # training and validation loop
        for epoch in xrange(args.num_epochs):
            # validate and display
            if epoch % args.valid_freq == 0:
                #tr_performance,tr_loss = tr_eval.evaluate(trainer.model)
                va_performance,va_loss = va_eval.evaluate(trainer.model)
                #tr_viz.update(tr_performance,tr_loss,epoch)
                #va_viz.update(va_performance,va_loss,epoch)
                # saves trained model by validation performance
                best_performance = save_model(args,
                                              trainer.model.state_dict(),
                                              va_performance,
                                              best_performance)

            # train
            total_loss = trainer.train_epoch()
            tp('d','Total training loss: ' + str(total_loss) + ' for epoch ' + str(epoch))
    else:
        # sets up for testing
        all_performance = []
        for fold in xrange(args.num_folds):
            tester,test_model = testing_setup(args,fold)
            te_performance,te_loss = tester.evaluate(test_model)
            all_performance.append(te_performance)
        print np.mean(np.asarray(all_performance),axis=0)

    tp('d','Total time:' + str(time()-start))

