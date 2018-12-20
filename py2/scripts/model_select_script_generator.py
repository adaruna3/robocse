lrs = [1e-1,1e-2,1e-3]
wdrs = [1e-2,1e-3,1e-4,1e-5]
bss = [50,100,200,400]
nds = [80,160,240,320]
command_root = "python run_robocse.py sd_thor tg_all_"
fixed_args = ' -m adagrad -bc 7'

with open('search_hyper_params.sh','w') as f:
    f.write("#!/bin/bash\n")
    count = 0
    for lr in lrs:
        for wdr in wdrs:
            for bs in bss:
                for nd in nds:
                    for fold in xrange(5):
                        exp_name = str(fold)+' -en '+str(count)
                        change_args = exp_name+' -p '+str(lr)+' 0 '+str(wdr)+\
                                      ' -bs '+str(bs)+' -d '+str(nd)
                        command = command_root + change_args + fixed_args
                        f.write(command+'\n')
                    count += 1