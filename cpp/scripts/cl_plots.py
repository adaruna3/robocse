import matplotlib.pyplot as plt
import numpy as np
from math import sqrt, isnan
import pdb


def get_entities():
    entities = []
    entity_map = {}
    entity_id = 0
    with open("../datasets/sd_thor_mp3d_entities.csv","r") as f:
        next(f)
        for line in f:
            entities.append(line.strip())
            entity_map[line.strip()] = entity_id
            entity_id += 1
    return entities, entity_map


def get_logged_performance_curves2(num_missing,method):
    entity_curves = {}
    file_root = "../robocse_logging/sd_thor_mp3d_tg_all_0_multE_"
    file_end = "_" + method + ".csv"
    gw, mw = get_wc_weights(num_missing)
    num_sets = 30
    for set_idx in xrange(num_sets):
        file_path = file_root + str(num_missing) + "_" + str(set_idx) + file_end
        with open(file_path,"r") as f:
            samples = np.ndarray(shape=(0,5))
            for line in f:
                sample = [[float(x) for x in line.strip().split(",")[:-1]]]
                sample[0].append(gw*sample[0][1]+mw*sample[0][2])
                samples = np.append(samples,sample,axis=0)
        entity_curves[set_idx] = samples
    return entity_curves


def get_average_num_epochs(curves,maxsize=151):
    average_num_epochs = 0.0
    average_divisor = 0.0
    for entity, samples in curves.items():
        if samples.shape[0] < maxsize:
            average_num_epochs += samples.shape[0]
            average_divisor += 1.0
    return average_num_epochs / average_divisor


def get_num_retrain_failures(curves,fail_length=100):
    num_failures = 0
    for entity, samples in curves.items():
        if samples.shape[0] == fail_length:
            num_failures += 1.0
    return num_failures


def get_target_values2(num_missing):
    target_values = {}
    file_root = "../trained_models/batch_learn/sd_thor_mp3d_tg_all_0_"
    file_end = ".txt"
    num_sets = 30
    for set_idx in xrange(num_sets):
        file_path = file_root + str(num_missing) + "_" + str(set_idx) + "_g" + file_end
        with open(file_path,"r") as f:
            g_target = float(f.readline().strip())
        file_path = file_root + str(num_missing) + "_" + str(set_idx) + "_m" + file_end
        with open(file_path,"r") as f:
            m_target = float(f.readline().strip())
        # gets the weighted combined value
        given_w, missing_w = get_wc_weights(num_missing)
        wc_target = given_w * g_target + missing_w * m_target
        target_values[set_idx] = [g_target,m_target,wc_target]
    return target_values

def get_target_value(num_missing):
    target_performance_book = get_target_values2(num_missing)
    g_target_avg = 0.0
    m_target_avg = 0.0
    m_target_div = 0.0
    wc_target_avg = 0.0
    wc_target_div = 0.0
    for entity, targets in target_performance_book.items():
        g_target_avg += targets[0]
        if not isnan(targets[1]):
            m_target_avg += targets[1]
            m_target_div += 1.0
        if not isnan(targets[2]):
            wc_target_avg += targets[2]
            wc_target_div += 1.0
    g_target_avg = g_target_avg / len(target_performance_book)
    m_target_avg = m_target_avg / m_target_div
    wc_target_avg = wc_target_avg / wc_target_div
    return g_target_avg, m_target_avg, wc_target_avg


def get_avg_num_epochs_specific(curves,target_values):
    g_average_num_epochs = 0.0
    m_average_num_epochs = 0.0
    diff_threshold = 0.1
    for entity, samples in curves.items():
        # missing num epochs
        num_epochs = 0.0
        m_target_value = target_values[entity][1]
        for m_sample_value in samples[:,2]:
            if m_target_value-m_sample_value < diff_threshold:
                break
            num_epochs += 1.0
        m_average_num_epochs += num_epochs
        # given num epochs
        num_epochs = 0.0
        g_target_value = target_values[entity][0]
        m_target_value = target_values[entity][1]
        for sample_idx in xrange(samples.shape[0]):
            g_sample_value = samples[sample_idx,1]
            m_sample_value = samples[sample_idx,2]
            g_diff = g_target_value-g_sample_value
            m_diff = m_target_value-m_sample_value
            if ( (m_diff < diff_threshold) and (g_diff < diff_threshold) ):
                break
            num_epochs += 1.0
        g_average_num_epochs += num_epochs

    g_average_num_epochs = g_average_num_epochs / len(curves)
    m_average_num_epochs = m_average_num_epochs / len(curves)
    return g_average_num_epochs, m_average_num_epochs

def get_num_retrain_failures_to_target(curves,target_values,fail_length=100):
    g_num_failures = 0.0
    m_num_failures = 0.0
    diff_threshold = 0.01
    for entity, samples in curves.items():
        # missing num failures
        num_epochs = 0.0
        m_target_value = target_values[entity][1]
        for m_sample_value in samples[:,2]:
            if m_target_value-m_sample_value < diff_threshold:
                break
            num_epochs += 1.0
        if (num_epochs >= fail_length):
            m_num_failures += 1.0
        # given num failures
        num_epochs = 0.0
        g_target_value = target_values[entity][0]
        m_target_value = target_values[entity][1]
        for sample_idx in xrange(samples.shape[0]):
            g_sample_value = samples[sample_idx,1]
            m_sample_value = samples[sample_idx,2]
            g_diff = g_target_value-g_sample_value
            m_diff = m_target_value-m_sample_value
            if ( (m_diff < diff_threshold) and (g_diff < diff_threshold) ):
                break
            num_epochs += 1.0
        if (num_epochs >= fail_length):
            g_num_failures += 1.0

    return g_num_failures, m_num_failures


def get_average_plot(curves,method,plot_curve=False):
    g_avg_curve = np.zeros(shape=(100))
    g_std_curve = np.zeros(shape=(100))
    m_avg_curve = np.zeros(shape=(100))
    m_std_curve = np.zeros(shape=(100))
    c_avg_curve = np.zeros(shape=(100))
    c_std_curve = np.zeros(shape=(100))
    wc_avg_curve = np.zeros(shape=(100))
    wc_std_curve = np.zeros(shape=(100))

    for curve_idx in xrange(g_avg_curve.shape[0]):
        num_samps = 0.0
        # get averages
        g_sum = 0.0
        m_sum = 0.0
        c_sum = 0.0
        wc_sum = 0.0
        nan_list = []
        for entity, samples in curves.items():
            if (curve_idx+1 <= samples.shape[0]):
                if not isnan(samples[curve_idx,2]):
                    g_sum += samples[curve_idx,1]
                    m_sum += samples[curve_idx,2]
                    c_sum += samples[curve_idx,3]
                    wc_sum += samples[curve_idx,4]
                    num_samps += 1.0
                else:
                    if not entity in nan_list:
                        nan_list.append(entity)
        if num_samps == 0:
            g_avg_curve[curve_idx] = float("Nan")
            m_avg_curve[curve_idx] = float("Nan")
            c_avg_curve[curve_idx] = float("Nan")
            wc_avg_curve[curve_idx] = float("Nan")
        else:
            g_avg_curve[curve_idx] = g_sum / num_samps
            m_avg_curve[curve_idx] = m_sum / num_samps
            c_avg_curve[curve_idx] = c_sum / num_samps
            wc_avg_curve[curve_idx] = wc_sum / num_samps

        # get stds
        g_sum = 0.0
        m_sum = 0.0
        c_sum = 0.0
        wc_sum = 0.0
        for entity, samples in curves.items():
            if curve_idx+1 <= samples.shape[0]:
                if not isnan(samples[curve_idx,2]):
                    g_sum += (samples[curve_idx,1]-g_avg_curve[curve_idx])**2
                    m_sum += (samples[curve_idx,2]-m_avg_curve[curve_idx])**2
                    c_sum += (samples[curve_idx,3]-c_avg_curve[curve_idx])**2
                    wc_sum += (samples[curve_idx,4]-wc_avg_curve[curve_idx])**2
        if num_samps == 0:
            g_std_curve[curve_idx] = float("Nan")
            m_std_curve[curve_idx] = float("Nan")
            c_std_curve[curve_idx] = float("Nan")
            wc_std_curve[curve_idx] = float("Nan")
        else:
            g_std_curve[curve_idx] = sqrt(g_sum / num_samps)
            m_std_curve[curve_idx] = sqrt(m_sum / num_samps)
            c_std_curve[curve_idx] = sqrt(c_sum / num_samps)
            wc_std_curve[curve_idx] = sqrt(wc_sum / num_samps)

    # plots the outputs
    #if plot_curve:
    #    epochs = np.asarray(range(0,100))
    #    fig, ax = plt.subplots()
    #    ax.plot(epochs, g_avg_curve, lw=2, label="mean over !E' triples", color='blue')
    #    ax.plot(epochs, m_avg_curve, lw=2, label="mean over E' triples", color='green')
    #    ax.fill_between(epochs, g_avg_curve+g_std_curve, g_avg_curve-g_std_curve, facecolor='blue', alpha=0.5)
    #    ax.fill_between(epochs, m_avg_curve+m_std_curve, m_avg_curve-m_std_curve, facecolor='green', alpha=0.5)
    #    ax.set_title(r"Performance over E' & !E' for " + method)
    #    ax.legend(loc='upper left')
    #    ax.set_xlabel('Epochs')
    #    ax.set_ylabel('Performance (AMRR %)')
    #    ax.grid()
    #    print("%s has NaNs" % str(nan_list))
    #    plt.savefig(method+".png")
    return g_avg_curve, m_avg_curve, c_avg_curve, wc_avg_curve

def get_comparison_plots(g_curves,m_curves,c_curves,wc_curves,target_performance_book,methods):
    colors = ['r','c','m','b','g']
    legend_labels = ['Xavier','Random','Similarity','Relational','Hybrid']
    g_target_avg = 0.0
    m_target_avg = 0.0
    m_target_div = 0.0
    wc_target_avg = 0.0
    wc_target_div = 0.0
    for entity, targets in target_performance_book.items():
        g_target_avg += targets[0]
        if not isnan(targets[1]):
            m_target_avg += targets[1]
            m_target_div += 1.0
        if not isnan(targets[2]):
            wc_target_avg += targets[2]
            wc_target_div += 1.0
    g_target_avg = g_target_avg / len(target_performance_book)
    m_target_avg = m_target_avg / m_target_div
    wc_target_avg = wc_target_avg / wc_target_div
    # make needed curves
    g_target_curve = np.ones(100)*g_target_avg
    m_target_curve = np.ones(100)*m_target_avg
    wc_target_curve = np.ones(100)*wc_target_avg
    print "WC Target Avg: " + str(wc_target_avg)
    print "G Target Avg: " + str(g_target_avg)
    epochs = np.asarray(range(0,100))
    # plots the outputs
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig4, ax4 = plt.subplots()
    ax1.plot(epochs, g_target_curve, lw=2, label="Expected Performance (Batch)", color='k')
    ax2.plot(epochs, m_target_curve, lw=2, label="Expected Performance (Batch)", color='k')
    ax4.plot(epochs, wc_target_curve, lw=2, label="Expected Performance (Batch)", color='k')
    for method_idx in xrange(len(methods)):
        ax1.plot(epochs, g_curves[method_idx], lw=2, label=legend_labels[method_idx], color=colors[method_idx])
        ax2.plot(epochs, m_curves[method_idx], lw=2, label=legend_labels[method_idx], color=colors[method_idx])
        ax4.plot(epochs, wc_curves[method_idx], lw=2, label=legend_labels[method_idx], color=colors[method_idx])
    ax1.set_title(r"Performance comparison over triples NOT related to inserted entites in E' set")
    ax2.set_title(r"Performance comparison over triples related to inserted entites in E' set")
    ax4.set_title(r"Performance comparison over all test triples (weighted)")
    ax1.set_ylim((0.5, 1.0))
    ax1.legend(loc='lower right')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Performance (AMRR)')
    ax1.grid()
    ax2.set_ylim((0.5, 1.0))
    ax2.legend(loc='lower right')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Performance (AMRR)')
    ax2.grid()
    ax4.set_ylim((0.5, 1.0))
    ax4.legend(loc='lower right')
    ax4.set_xlabel('Epochs')
    ax4.set_ylabel('Performance (AMRR)')
    ax4.grid()
    fig1.savefig("given_comparison.png")
    fig2.savefig("missing_comparison.png")
    fig4.savefig("weighted_comparison.png")
    plt.show()

def get_individual_num_missing_plots(num_missing_ents):
    insertion_methods = ['xavier','random','similarity','relational_node','hybrid']
    converge_targets = get_target_values2(num_missing_ents)
    m_avg_curves = []
    g_avg_curves = []
    c_avg_curves = []
    wc_avg_curves = []
    for insertion_method_idx in xrange(len(insertion_methods)):
        insertion_method = insertion_methods[insertion_method_idx]
        method_curves = get_logged_performance_curves2(num_missing_ents,insertion_method)
        all_avg_num_epochs = get_average_num_epochs(method_curves)
        g_avg_num_epochs, m_avg_num_epochs = get_avg_num_epochs_specific(method_curves,converge_targets)
        print("Average number of epochs for %s, all: %4.2f, !E': %4.2f E': %4.2f" % (insertion_method,all_avg_num_epochs,g_avg_num_epochs,m_avg_num_epochs))
        all_num_failures = get_num_retrain_failures(method_curves)
        g_num_failures, m_num_failures = get_num_retrain_failures_to_target(method_curves,converge_targets)
        print("Number of failures for %s, all: %4.2f, !E': %4.2f E': %4.2f" % (insertion_method,all_num_failures,g_num_failures,m_num_failures))
        g_avg, m_avg, c_avg, wc_avg = get_average_plot(method_curves,insertion_method,True)
        g_avg_curves.append(g_avg)
        m_avg_curves.append(m_avg)
        c_avg_curves.append(c_avg)
        wc_avg_curves.append(wc_avg)
    make_csv('mconvergence.csv',insertion_methods,m_avg_curves)
    make_csv('gconvergence.csv',insertion_methods,g_avg_curves)
    make_csv('cconvergence.csv',insertion_methods,c_avg_curves)
    make_csv('wcconvergence.csv',insertion_methods,wc_avg_curves)
    get_comparison_plots(g_avg_curves,m_avg_curves,c_avg_curves,wc_avg_curves,converge_targets,insertion_methods)


def get_num_epochs_spared(curve,threshold):
    num_epochs = -1.0
    for epoch in xrange(len(curve)):
        if curve[epoch] >= threshold:
            num_epochs = 100.0 - epoch
            break
    if num_epochs == -1.0:
        num_epochs = 0.0
    return num_epochs

def get_num_missing_efficiency_comparison_plots():
    insertion_methods = ['xavier','random','similarity','relational_node','hybrid']
    all_num_missing_ents = [1,2,3,4,5,6,7,8,9,10]
    wc_num_epochs_cruves = []
    for insertion_method_idx in xrange(len(insertion_methods)):
        insertion_method = insertion_methods[insertion_method_idx]
        wc_num_epochs_curve = []
        for num_missing_ents in all_num_missing_ents:
            if num_missing_ents == -1.0:
                wc_num_epochs_curve.append(float('Nan'))
                continue
            method_curves = get_logged_performance_curves2(num_missing_ents,insertion_method)
            g_avg, m_avg, c_avg, wc_avg = get_average_plot(method_curves,insertion_method)
            g_target, m_target, wc_target = get_target_value(num_missing_ents)
            wc_num_epochs_curve.append(get_num_epochs_spared(wc_avg,wc_target-0.1))
        wc_num_epochs_cruves.append(np.asarray(wc_num_epochs_curve))
    plot_efficiency_comparison(wc_num_epochs_cruves)


def plot_efficiency_comparison(curves):
    insertion_methods = ['xavier','random','similarity','relational_node','hybrid']
    colors = ['r','c','m','b','g']
    markers = ['D','o','+','x','*']
    labels = ['Xavier','Random','Similarity','Relational','Hybrid']

    # plots the outputs
    num_missing_ents = np.asarray([1,2,3,4,5,6,7,8,9,10])
    fig1, ax1 = plt.subplots() # num epochs
    linestyle_flag = 'None'
    for method_idx in xrange(len(insertion_methods)):
        ax1.plot(num_missing_ents, curves[method_idx], lw=2, label=labels[method_idx], marker=markers[method_idx], color=colors[method_idx],linestyle=linestyle_flag)
    ax1.set_title(r"Efficiency (100-Number Epochs) for E' Inference to Converge")
    ax1.set_ylim((0.0, 100.0))
    ax1.legend(loc='lower right')
    ax1.set_xlabel('Number of OOKB entities')
    ax1.set_ylabel('Efficiency (100 - Required Number of Epochs to Converge)')
    ax1.grid()
    fig1.savefig("efficiency_missing_comparison.png")
    make_csv("efficiency_missing_comparison.csv",insertion_methods,curves)
    #plt.show()


def get_logged_avg(fp,metrix_idx):
    #root = "../robocse_logging/eta_0p001_gamma_0p001/"
    root = "../robocse_logging/"
    #root = "../robocse_logging/eta_0p002_gamma_0p1/"
    weights = [4312.0,2963.0,7484.0,14759.0]
    raw_values = []
    with open(root+fp,'r') as f:
        for line in f:
            raw_values.append(float(line.strip().split(",")[metrix_idx]))
    avg_values = []
    for i in xrange(len(raw_values)/2):
        avg_values.append((raw_values[i]+raw_values[i+4])/2.0)
    avg_values.append(raw_values[3])
    w_avg = 0.0
    normalizer = 0.0
    for avg_value_idx in xrange(len(avg_values)):
        if isnan(avg_values[avg_value_idx]):
            continue
        w_avg += avg_values[avg_value_idx]*weights[avg_value_idx]
        normalizer += weights[avg_value_idx]
    w_avg = w_avg / normalizer
    return w_avg

def get_wc_from_logged_avg(nme,method,metric_idx):
    root = "../robocse_logging/"
    # gets given avg values
    fp = "immediate_given_avg_" + str(nme) + "_" + method + ".csv"
    immediate_given_avg = get_logged_avg(root+fp,metric_idx)
    fp = "final_given_avg_" + str(nme) + "_" + method + ".csv"
    final_given_avg = get_logged_avg(root+fp,metric_idx)
    # gets missing avg values
    fp = "immediate_missing_avg_" + str(nme) + "_" + method + ".csv"
    immediate_missing_avg = get_logged_avg(root+fp,metric_idx)
    fp = "final_missing_avg_" + str(nme) + "_" + method + ".csv"
    final_missing_avg = get_logged_avg(root+fp,metric_idx)
    # gets weights for given and missing
    given_weight, missing_weight = get_wc_weights(nme)
    # calculates the weighted combined values
    immediate_wc_value = immediate_given_avg * given_weight + immediate_missing_avg * missing_weight
    final_wc_value = final_given_avg * given_weight + final_missing_avg * missing_weight
    return immediate_wc_value, final_wc_value

def get_wc_weights(nme):
    ents, emap = get_entities()
    file_root = "../trained_models/missing_ent_files/sd_thor_mp3d_tg_all_0_ments_"
    nms = 30
    missing_avg_num_triples = 0
    given_avg_num_triples = 0
    for missing_set_idx in xrange(nms):
        # gets the missing ents
        fp = file_root + str(nme) + "_" + str(missing_set_idx) + ".csv"
        missing_ents = []
        with open(fp,'r') as f:
            for line in f:
                missing_ents.append(int(line.strip()))
        # counts number of triples in given, not including missing ents
        fp = "../datasets/sd_thor_mp3d_tg_all_0_test.csv"
        with open(fp,'r') as f:
            next(f)
            for line in f:
                subj,rel,obj = line.strip().split(',')
                if emap[subj] in missing_ents or emap[obj] in missing_ents:
                    continue
                else:
                    given_avg_num_triples += 1
        # counts number of unique triples in missing
        fp = "../datasets/sd_thor_mp3d_tg_all_0_gt.csv"
        unique_triples = []
        with open(fp,'r') as f:
            next(f)
            for line in f:
                subj,rel,obj = line.strip().split(',')
                if emap[subj] in missing_ents or emap[obj] in missing_ents:
                    if line.strip() not in unique_triples:
                        unique_triples.append(line.strip())
                else:
                    continue
            missing_avg_num_triples += len(unique_triples)

    # averages number of given and missing triples across all missing sets
    missing_avg_num_triples = float(missing_avg_num_triples)/float(nms)
    given_avg_num_triples = float(given_avg_num_triples)/float(nms)
    # calculates the weights
    missing_weight = missing_avg_num_triples / (missing_avg_num_triples + given_avg_num_triples)
    given_weight = given_avg_num_triples / (missing_avg_num_triples + given_avg_num_triples)
    return given_weight, missing_weight


def make_csv(fp,methods,curves):
    with open(fp,'w') as f:
        for method_idx in range(len(methods)):
            method = methods[method_idx]
            if method_idx < len(methods)-1:
                f.write(method+',')
            else:
                f.write(method)
        f.write('\n')
        for curve_idx in xrange(curves[0].shape[0]):
            for method_idx in range(len(methods)):
                if method_idx < len(methods)-1:
                    f.write(str(curves[method_idx][curve_idx])+',')
                else:
                    f.write(str(curves[method_idx][curve_idx]))
            f.write('\n')


def get_num_missing_performance_comparison_plots(save):
    insertion_methods = ['xavier','random','similarity','relational_node','hybrid','batch']
    metric = {"mrr":0,"hits10":1,"hits5":2,"hits1":3,"map":4}
    colors = ['r','c','m','b','g','k']
    markers = ['D','o','+','x','*','+']
    labels = ['Xavier','Random','Similarity','Relational','Hybrid','Expected (Batch)']
    immediate_g_curves = []
    immediate_m_curves = []
    immediate_wc_curves = []
    final_g_curves = []
    final_m_curves = []
    final_wc_curves = []
    all_num_missing_ents = [-1,-2,-3,-4,-5,-6,-7,-8,-9,10]
    for insertion_method_idx in xrange(len(insertion_methods)):
        insertion_method = insertion_methods[insertion_method_idx]
        immediate_g_curve = np.zeros(shape=(10))
        immediate_m_curve = np.zeros(shape=(10))
        immediate_wc_curve = np.zeros(shape=(10))
        final_g_curve = np.zeros(shape=(10))
        final_m_curve = np.zeros(shape=(10))
        final_wc_curve = np.zeros(shape=(10))
        for num_missing_ents in all_num_missing_ents:
            if num_missing_ents < 0:
                immediate_g_curve[num_missing_ents-1] = float('Nan')
                immediate_m_curve[num_missing_ents-1] = float('Nan')
                immediate_wc_curve[num_missing_ents-1] = float('Nan')
                final_g_curve[num_missing_ents-1] = float('Nan')
                final_m_curve[num_missing_ents-1] = float('Nan')
                final_wc_curve[num_missing_ents-1] = float('Nan')
            else:
                if insertion_method == 'batch':
                    converge_targets = get_target_values2(num_missing_ents)
                    g_target_avg = 0.0
                    m_target_avg = 0.0
                    m_target_div = 0.0
                    wc_target_avg = 0.0
                    wc_target_div = 0.0
                    for entity, targets in converge_targets.items():
                        g_target_avg += targets[0]
                        if not isnan(targets[1]):
                            m_target_avg += targets[1]
                            m_target_div += 1.0
                        if not isnan(targets[2]):
                            wc_target_avg += targets[2]
                            wc_target_div += 1.0
                    g_target_avg = g_target_avg / len(converge_targets)
                    m_target_avg = m_target_avg / m_target_div
                    wc_target_avg = wc_target_avg / wc_target_div

                    immediate_g_curve[num_missing_ents-1] = g_target_avg*100.0
                    immediate_m_curve[num_missing_ents-1] = m_target_avg*100.0
                    immediate_wc_curve[num_missing_ents-1] = wc_target_avg*100.0
                    final_g_curve[num_missing_ents-1] = g_target_avg*100.0
                    final_m_curve[num_missing_ents-1] = m_target_avg*100.0
                    final_wc_curve[num_missing_ents-1] = wc_target_avg*100.0
                    continue
                else:
                    fp = "immediate_given_avg_" + str(num_missing_ents) + "_" + insertion_method + ".csv"
                    value = get_logged_avg(fp,metric["mrr"])
                    immediate_g_curve[num_missing_ents-1] = value

                    fp = "immediate_missing_avg_" + str(num_missing_ents) + "_" + insertion_method + ".csv"
                    value = get_logged_avg(fp,metric["mrr"])
                    immediate_m_curve[num_missing_ents-1] = value

                    fp = "final_given_avg_" + str(num_missing_ents) + "_" + insertion_method + ".csv"
                    value = get_logged_avg(fp,metric["mrr"])
                    final_g_curve[num_missing_ents-1] = value

                    fp = "final_missing_avg_" + str(num_missing_ents) + "_" + insertion_method + ".csv"
                    value = get_logged_avg(fp,metric["mrr"])
                    final_m_curve[num_missing_ents-1] = value

                    # gets the weighted combined results
                    immediate_wc_value, final_wc_value = get_wc_from_logged_avg(num_missing_ents,insertion_method,metric["mrr"])
                    immediate_wc_curve[num_missing_ents-1] = immediate_wc_value
                    final_wc_curve[num_missing_ents-1] = final_wc_value

        immediate_g_curves.append(immediate_g_curve)
        immediate_m_curves.append(immediate_m_curve)
        immediate_wc_curves.append(immediate_wc_curve)
        final_g_curves.append(final_g_curve)
        final_m_curves.append(final_m_curve)
        final_wc_curves.append(final_wc_curve)

    # plots the outputs
    num_missing_ents = np.asarray(range(1,11))
    fig1, ax1 = plt.subplots() # immediate given
    fig2, ax2 = plt.subplots() # immediate missing
    fig7, ax7 = plt.subplots() # immediate weighted combined
    fig3, ax3 = plt.subplots() # final given
    fig4, ax4 = plt.subplots() # final missing
    fig8, ax8 = plt.subplots() # final weighted combined
    linestyle_flag = 'None'
    for method_idx in xrange(len(insertion_methods)):
        ax1.plot(num_missing_ents, immediate_g_curves[method_idx], lw=2, label=labels[method_idx], marker=markers[method_idx], color=colors[method_idx],linestyle=linestyle_flag)
        ax2.plot(num_missing_ents, immediate_m_curves[method_idx], lw=2, label=labels[method_idx], marker=markers[method_idx], color=colors[method_idx],linestyle=linestyle_flag)
        ax7.plot(num_missing_ents, immediate_wc_curves[method_idx], lw=2, label=labels[method_idx], marker=markers[method_idx], color=colors[method_idx],linestyle=linestyle_flag)
        ax3.plot(num_missing_ents, final_g_curves[method_idx], lw=2, label=labels[method_idx], marker=markers[method_idx], color=colors[method_idx],linestyle=linestyle_flag)
        ax4.plot(num_missing_ents, final_m_curves[method_idx], lw=2, label=labels[method_idx], marker=markers[method_idx], color=colors[method_idx],linestyle=linestyle_flag)
        ax8.plot(num_missing_ents, final_wc_curves[method_idx], lw=2, label=labels[method_idx], marker=markers[method_idx], color=colors[method_idx],linestyle=linestyle_flag)

    ax1.set_title(r"Immediate Performance over triples NOT related to OOKB entites (E)")
    ax2.set_title(r"Immediate Performance over triples related to OOKB entites (E')")
    ax7.set_title(r"Immediate Performance over all triples (weighted)")
    ax3.set_title(r"Final Performance over triples NOT related to OOKB entites (E)")
    ax4.set_title(r"Final Performance over triples related to OOKB entites (E')")
    ax8.set_title(r"Final Performance over all triples (weighted)")
    ax1.set_ylim((0.0, 100.0))
    ax1.legend(loc='lower right')
    ax1.set_xlabel('Number of OOKB entities')
    ax1.set_ylabel('Performance (AMRR)')
    ax1.grid()
    ax2.set_ylim((0.0, 100.0))
    ax2.legend(loc='lower right')
    ax2.set_xlabel('Number of OOKB entities')
    ax2.set_ylabel('Performance (AMRR)')
    ax2.grid()
    ax3.set_ylim((0.0, 100.0))
    ax3.legend(loc='lower right')
    ax3.set_xlabel('Number of OOKB entities')
    ax3.set_ylabel('Performance (AMRR)')
    ax3.grid()
    ax4.set_ylim((0.0, 100.0))
    ax4.legend(loc='lower right')
    ax4.set_xlabel('Number of OOKB entities')
    ax4.set_ylabel('Performance (AMRR)')
    ax4.grid()
    ax7.set_ylim((0.0, 100.0))
    ax7.legend(loc='lower right')
    ax7.set_xlabel('Number of OOKB entities')
    ax7.set_ylabel('Performance (AMRR)')
    ax7.grid()
    ax8.set_ylim((0.0, 100.0))
    ax8.legend(loc='lower right')
    ax8.set_xlabel('Number of OOKB entities')
    ax8.set_ylabel('Performance (AMRR)')
    ax8.grid()
    if save:
        fig1.savefig("immediate_given_comparison.png")
        fig2.savefig("immediate_missing_comparison.png")
        fig3.savefig("final_given_comparison.png")
        fig4.savefig("final_missing_comparison.png")
        fig7.savefig("immediate_weighted_comparison.png")
        fig8.savefig("final_weighted_comparison.png")
        make_csv("immediate_given_comparison.csv",insertion_methods,immediate_g_curves)
        make_csv("immediate_missing_comparison.csv",insertion_methods,immediate_m_curves)
        make_csv("immediate_weighted_comparison.csv",insertion_methods,immediate_wc_curves)
        make_csv("final_given_comparison.csv",insertion_methods,final_g_curves)
        make_csv("final_missing_comparison.csv",insertion_methods,final_m_curves)
        make_csv("final_weighted_comparison.csv",insertion_methods,final_wc_curves)
    #plt.show()

def get_immediate_mrr_curves(curves):
  g_imm_mrr = np.zeros(shape=(0))
  m_imm_mrr = np.zeros(shape=(0))
  c_imm_mrr = np.zeros(shape=(0))
  wc_imm_mrr = np.zeros(shape=(0))

  for entity_set, samples in curves.items():
      g_imm_mrr = np.append(g_imm_mrr, [samples[0,1]], 0)
      m_imm_mrr = np.append(m_imm_mrr, [samples[0,2]], 0)
      c_imm_mrr = np.append(c_imm_mrr, [samples[0,3]], 0)
      wc_imm_mrr = np.append(wc_imm_mrr, [samples[0,4]], 0)

  return g_imm_mrr, m_imm_mrr, c_imm_mrr, wc_imm_mrr

def get_immediate_mrr_wc_csvs():
    insertion_methods = ['xavier','random','similarity','relational_node','hybrid']
    all_num_missing_ents = [1,2,3,4,5,6,7,8,9,10]
    imm_wc_mrr_curves = []
    for num_missing_ents in all_num_missing_ents:
        for insertion_method_idx in xrange(len(insertion_methods)):
            insertion_method = insertion_methods[insertion_method_idx]
            method_curves = get_logged_performance_curves2(num_missing_ents,insertion_method)
            _,_,_, imm_wc_mrr_curve = get_immediate_mrr_curves(method_curves)
            imm_wc_mrr_curves.append(imm_wc_mrr_curve)
        fp = 'imm_wc_mrr_' + str(num_missing_ents) + '.csv'
        make_csv(fp,insertion_methods,imm_wc_mrr_curves)
        imm_wc_mrr_curves = []

if __name__ == "__main__":
    get_individual_num_missing_plots(10)
    #get_num_missing_efficiency_comparison_plots()
    #get_num_missing_performance_comparison_plots(True)
    #get_immediate_mrr_wc_csvs()