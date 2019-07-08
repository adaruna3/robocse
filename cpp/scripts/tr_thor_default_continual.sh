#!/bin/bash
####################
#
# RoboCSE Default Training
#
####################
#./opt_rcse_continual -dataset sd_thor -experiment tg_all_0 -num_thread 8 -ins_method relational
#./opt_rcse_continual -dataset sd_thor -experiment tg_all_0 -num_thread 8 -ins_method random
#./opt_rcse_continual -dataset sd_thor -experiment tg_all_0 -num_thread 8 -ins_method similarity
#./opt_rcse_continual -dataset sd_thor -experiment tg_all_0 -num_thread 8 -ins_method relational_node
./opt_rcse_continual -dataset sd_thor -experiment tg_all_0 -num_thread 8 -ins_method hybrid