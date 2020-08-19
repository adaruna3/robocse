import os
from torch.utils.tensorboard import SummaryWriter

import pdb


class ProcessorViz:
    def __init__(self, args):
        log_name = str(args.tag) + "__"
        log_name += str(args.dataset) + "_"
        log_name += "mt" + str(args.model) + "_"
        log_name += "clm" + str(args.cl_method)

        log_dir = os.path.abspath(os.path.dirname(__file__)) + "/logs/"
        self.log_fp = log_dir + log_name
        if os.path.isdir(self.log_fp):  # overwrites existing events log
            files = os.listdir(self.log_fp)
            for filename in files:
                if "events" in filename:
                    os.remove(self.log_fp+"/"+filename)
                # rmtree(self.log_fp)
        self._writer = SummaryWriter(self.log_fp)
        self.timestamp = 0
        self.gruvae_timestamp = 0

    def add_tr_sample(self, sess, sample):
        loss = sample
        self._writer.add_scalar("Loss/TrainSess_"+str(sess), loss, self.timestamp)
        self.timestamp += 1

    def add_de_sample(self, sample):
        hits_avg = 0.0
        mrr_avg = 0.0
        for sess in range(sample.shape[0]):
            hits, mrr = sample[sess,:]
            self._writer.add_scalar("HITS/DevSess_"+str(sess), hits, self.timestamp)
            self._writer.add_scalar("MRR/DevSess_"+str(sess), mrr, self.timestamp)
            hits_avg += hits
            mrr_avg += mrr
        hits_avg = hits_avg / float(sample.shape[0])
        mrr_avg = mrr_avg / float(sample.shape[0])
        self._writer.add_scalar("HITS/DevAvg", hits_avg, self.timestamp)
        self._writer.add_scalar("MRR/DevAvg", mrr_avg, self.timestamp)
