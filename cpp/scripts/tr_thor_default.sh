#!/bin/bash
####################
#
# RoboCSE Default Training
#
####################
./opt_main -dataset sd_thor -experiment tg_all_0 -num_thread 8
./opt_main -dataset sd_thor -experiment tg_all_1 -num_thread 8
./opt_main -dataset sd_thor -experiment tg_all_2 -num_thread 8
./opt_main -dataset sd_thor -experiment tg_all_3 -num_thread 8
./opt_main -dataset sd_thor -experiment tg_all_4 -num_thread 8