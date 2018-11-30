#!/bin/bash
####################
#
# RoboCSE Default Training
#
####################
python run_robocse.py sd_thor tg_all_0 -m adagrad -p 0.1 0.0 1e-3 -c 7
python run_robocse.py sd_thor tg_all_1 -m adagrad -p 0.1 0.0 1e-3 -c 7
python run_robocse.py sd_thor tg_all_2 -m adagrad -p 0.1 0.0 1e-3 -c 7
python run_robocse.py sd_thor tg_all_3 -m adagrad -p 0.1 0.0 1e-3 -c 7
python run_robocse.py sd_thor tg_all_4 -m adagrad -p 0.1 0.0 1e-3 -c 7
