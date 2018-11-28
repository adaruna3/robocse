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
    def __init__(self,legend,amrr_idx=0,hits_idx=2):
        # connect to the Visdom server
        self.viz = Visdom()
        assert self.viz.check_connection(),'Start Visdom!'
        # clears main environment
        self.viz.close()
        # initialize the visualization arrays
        self.hits = np.empty(shape=(3,len(legend),0))
        self.amrr = np.empty(shape=(3,len(legend),0))
        self.loss = np.empty(shape=(1,0))
        self.amrr_idx = amrr_idx
        self.hits_idx = hits_idx
        # plot the initial metrics and loss
        self.wins = []
        self.initial_update = True
        self.legend = legend

    def update(self,metric,loss):
        append_shape = (self.hits.shape[0],self.hits.shape[1],1)
        hits_append = metric[:,:,self.hits_idx].reshape(append_shape)
        amrr_append = metric[:,:,self.amrr_idx].reshape(append_shape)
        loss_append = np.asarray(loss).reshape((1,1))
        # store logged values
        self.hits = np.append(self.hits,hits_append,axis=2)
        self.amrr = np.append(self.amrr,amrr_append,axis=2)
        self.loss = np.append(self.loss,loss_append,axis=1)
        append_idx = self.loss.shape[1]
        # initial update
        if self.initial_update:
            # initialize windows and plot
            self.initial_update = False
            title_ending = {0:"Sro",1:"sRo",2:"srO"}
            for i in xrange(7):
                if i < 3:
                    x_axis = np.full_like(amrr_append[i].T,append_idx)
                    title = 'AMRR ' + title_ending[i]
                    self.wins.append(self.viz.line(X=x_axis,
                                                   Y=amrr_append[i].T,
                                                   opts=dict(title=title),))
                elif i < 4:
                    x_axis = np.full_like(loss_append.T,append_idx)
                    title = 'Total Epoch Loss'
                    self.wins.append(self.viz.line(X=x_axis,
                                                   Y=loss_append.T,
                                                   opts=dict(title=title),))
                else:
                    x_axis = np.full_like(hits_append[(i+2)%3].T,append_idx)
                    title = 'Hits@5 ' + title_ending[(i+2)%3]
                    self.wins.append(self.viz.line(X=x_axis,
                                                   Y=hits_append[(i+2)%3].T,
                                                   opts=dict(title=title),))
        else:
            # update plots
            for i in xrange(7):
                if i < 3:
                    opts = dict(legend=self.legend)
                    self.viz.line(X=np.full_like(amrr_append[i].T,append_idx),
                                  Y=amrr_append[i].T,
                                  win=self.wins[i],
                                  update='append',
                                  opts=opts)
                elif i < 4:
                    opts = dict(legend=self.legend)
                    self.viz.line(X=np.full_like(loss_append.T,append_idx),
                                  Y=loss_append.T,
                                  win=self.wins[i],
                                  update='append')
                else:
                    opts = dict(legend=self.legend)
                    self.viz.line(X=np.full_like(hits_append[(i+2)%3].T,append_idx),
                                  Y=hits_append[(i+2)%3].T,
                                  win=self.wins[i],
                                  update='append',
                                  opts=opts)

if __name__ == "__main__":
    from time import sleep
    import pdb

    number_of_relation_types = 3
    tr_viz = RoboCSETrainViz(['Loc','Mat','Aff'])
    for i in xrange(3):
        tr_viz.update(np.random.uniform(size=(3,number_of_relation_types,4)),
                      np.random.randint(10))
        sleep(3)