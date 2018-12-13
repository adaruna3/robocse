#!/bin/bash
####################
#
# RoboCSE Default Training
#
####################
python run_robocse.py sd_thor tg_all_0 -m adagrad -p 1e-2 0 1e-4 -bc 7 -bs 200 -d 200
python run_robocse.py sd_thor tg_all_1 -m adagrad -p 1e-2 0 1e-4 -bc 7 -bs 200 -d 200
python run_robocse.py sd_thor tg_all_2 -m adagrad -p 1e-2 0 1e-4 -bc 7 -bs 200 -d 200
python run_robocse.py sd_thor tg_all_3 -m adagrad -p 1e-2 0 1e-4 -bc 7 -bs 200 -d 200
python run_robocse.py sd_thor tg_all_4 -m adagrad -p 1e-2 0 1e-4 -bc 7 -bs 200 -d 200