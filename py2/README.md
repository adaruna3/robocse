# Py-RoboCSE

## Overview
Repo implements basic functionality to train and test "RoboCSE" models.

## Required Files
In order to train/test models, please provide a `datasets` folder in this directory contiaining a `[dataset name]_entites.csv` file, `[dataset name]_relations.csv` file, and at least one set of `[dataset name]_[experiment_name]_gt.csv`,`[dataset name]_[experiment_name]_train.csv`,`[dataset name]_[experiment_name]_valid.csv`,`[dataset name]_[experiment_name]_test.csv` files. 

The entities file provides a list of all entities and relations file a list of all relations. The train, valid, and test files provide triples to be used for training, validation, and testing, respectively. The ground-truth file (gt) is the set of all triples (i.e. across the single train, valid, and test set).

An example `datasets` directory that can be copy/pasted in in the [root](https://github.com/adaruna3/RoboCSE) of this repository. See those files for format. Note that this dataset contains multiple sets of train, valid, and test files because each set is part of a 5-fold-cross-validation.

**Note for Py-RoboCSE**: When copying datasets folder, add a `__init__.py` file inside the folder so the folder is recognized as a module by Python.

## Training Py-RoboCSE
1. Start the visdom server for visdom visualizations via `visdom`.
2. You can train a single fold from the examples `datasets` directory via following command from the `tr_thor_default.sh` script in the `scripts` folder: `python run_robocse.py sd_thor tg_all_0 -m adagrad -p 1e-2 0 1e-4 -bc 7 -bs 200 -d 200`.
3. While training, evaluations of the valid triples will be visualized in visdom plots. You can view these plots in a web-browser at `http://localhost:8097/`.

## Testing C++ RoboCSE (5-fold-cross-validation)
1. First train a model for each fold of the model by running `./scripts/tr_thor_default.sh`. As models are trained/saved they will appear in the `trained_models` folder.
2. Once a model for each fold has been trained, you can test across all folds via `python run_robocse.py sd_thor tg_all -d 100 -t 0`.
3. Test evaluation will print to terminal.
