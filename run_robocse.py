# torch imports
from torch import save

# RoboCSE imports
from trvate_utils import training_setup,validation_setup
from robocse_logging.viz_utils import tp,valid_visualization_setup
import trained_models

# system imports
import pdb
from os.path import abspath,dirname
import argparse
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
    parser.add_argument('-e', dest='valid_freq', type=int, default=10,
                        nargs='?', help='Evaluation frequency')
    parser.add_argument('-t', dest='train', type=int, default=1,
                        nargs='?', help='Train=1,Test=0')
    parser.add_argument('-c', dest='batch_cutoff', type=int, default=None,
                        nargs='?', help='Cutoff for training visualization')
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


def save_model(dataset_name,experiment_name,params):
    models_fp = abspath(dirname(trained_models.__file__)) + '/'
    model_fp = models_fp + dataset_name + '_' + experiment_name + '.pt'
    save(params,model_fp)
    tp('s','model for experiment ' + experiment_name + ' on ' + dataset_name + \
       'trained and saved successfully.')
    tp('s','Written to: '+model_fp)

if __name__ == "__main__":
    # parses command line arguments
    args = parse_command_line()

    if args.train:
        # sets up for training
        trainer = training_setup(args)
        # sets up for validation
        tr_eval,va_eval = validation_setup(args)
        # sets up for visualizing training/validation
        tr_viz,va_viz = valid_visualization_setup(tr_eval,va_eval)

        # training and validation loop
        for epoch in xrange(args.num_epochs):
            # validate and display
            if epoch % args.valid_freq == 0:
                tr_performance,tr_loss = tr_eval.evaluate(trainer.model)
                va_performance,va_loss = va_eval.evaluate(trainer.model)

                tr_viz.update(tr_performance,tr_loss)
                va_viz.update(va_performance,va_loss)

            # train
            total_loss = trainer.train_epoch()
            tp('d','Total training loss: ' + str(total_loss) + ' for epoch ' + str(epoch))

        # save the trained model
        save_model(args.ds_name,args.exp_name,trainer.model.state_dict())