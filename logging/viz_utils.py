from visdom import Visdom
import numpy as np


type2color = {
    's': ' \033[95mSuccess:\033[0m ',
    'i': ' \033[94mInfo:\033[0m ',
    'd': ' \033[92mDebug:\033[0m ',
    'w': ' \033[93mWarning:\033[0m ',
    'e': ' \033[91mError:\033[0m ',
    'f': ' \033[4m\033[1m\033[91mFatal Error:\033[0m '
}

# terminal printing (tp)
def tp(p_type,msg,update=0):
    if not p_type.lower() in type2color:
        if update == 1:
            print '\r'+msg,
        else:
            print msg
    else:
        start = type2color[p_type.lower()]
        if update == 1:
            print '\r'+start+msg,
        else:
            print start + msg

class RoboCSETrainViz():
    def __init__(self,num_rels,amrr_idx=0,hits_idx=2):
        # connect to the Visdom server
        self.viz = Visdom()
        assert self.viz.check_connection()
        # initialize the visualization arrays
        self.hits = np.zeros(shape=(3,num_rels,1))
        self.amrr = np.zeros(shape=(3,num_rels,1))
        self.loss = np.zeros(shape=(1,1))
        self.amrr_idx = amrr_idx
        self.hits_idx = hits_idx
        # plot the initial metrics and loss
        self.windows = []
        for i in xrange(7):
            if i < 3:
                self.windows.append(self.viz.line(Y=self.amrr[i],
                                                  opts=dict(title='AMRR'),))
            elif i < 6:
                self.windows.append(self.viz.line(Y=self.hits[i%3],
                                                  opts=dict(title='Hits@5'),))
            else:
                self.windows.append(self.viz.line(Y=self.loss,
                                                  opts=dict(title='Loss'),))

    def update(self,metric,loss):
        raise NotImplementedError

if __name__ == "__main__":
    tr_viz = RoboCSETrainViz(3)

