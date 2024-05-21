'''
@File  :calibration_metrics.py
@Date  :2023/1/27 17:23
@Desc  :
'''

import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as mtick

def plot_acc_calibration(path, output_softmax, ground_truth, acc, ece, n_bins = 15, title = None, epoch=None):
    p_value, pred_label= torch.max(output_softmax, 1)
    confidence_all, confidence_acc = np.zeros(n_bins), np.zeros(n_bins)

    sub_n_bins = n_bins * 2
    bins = np.arange(0, 1.0 + 1 / sub_n_bins-0.0001, 1 / sub_n_bins)
    sub_weights = np.ones(len(ground_truth)) / float(len(ground_truth))
    sub_acc = np.zeros_like(ground_truth)
    for index, value in enumerate(p_value):
        #value -= suboptimal_prob[index]
        interval = int(value / (1 / n_bins) -0.0001)
        confidence_all[interval] += 1
        if pred_label[index] == ground_truth[index]:
            confidence_acc[interval] += 1
    for index, value in enumerate(confidence_acc):
        if confidence_all[index] == 0:
            confidence_acc[index] = 0
        else:
            confidence_acc[index] /= confidence_all[index]
    for index, value in enumerate(p_value):
        interval = int(value / (1 / n_bins) - 0.0001)
        sub_acc[index]=confidence_acc[interval]

    start = np.around(1/n_bins/2, 3)
    step = np.around(1/n_bins, 3)
    # plt.figure(figsize=(6, 5))
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.bar(np.around(np.arange(start, 1.0, step), 3), confidence_acc,
           alpha=0.7, width=0.05, color='dodgerblue', label='Outputs')
    ax.bar(np.around(np.arange(start, 1.0, step), 3),
            np.around(np.arange(start, 1.0, step), 3), alpha=0.7, width=0.05, color='orange', label='Expected')


    # ax.set_aspect(1.)
    ax.plot([0,1], [0,1], ls='--',c='k')
    divider = make_axes_locatable(ax)
    ax_histx = divider.append_axes("top", 0.6, pad=0.3, sharex=ax)
    ax_histy = divider.append_axes("right", 0.6, pad=0.3, sharey=ax)
    ax_histx.xaxis.set_tick_params(labelbottom=False)
    ax_histy.yaxis.set_tick_params(labelleft=False)

    ax_histx.hist(p_value, bins=bins, edgecolor='white',
                  color='lightblue', weights=sub_weights)
    ax_histy.hist(sub_acc, bins=bins, orientation='horizontal',
                  edgecolor='white', color='lightblue', weights=sub_weights)
    ax_histx.plot([p_value.mean().tolist(), p_value.mean().tolist()], [0, 1], ls='-', c='r', linewidth=3)
    ax_histy.plot([0, 1], [acc, acc], ls='-', c='r', linewidth=3)

    ax_histx.set_yticks([0, 0.5, 1])
    ax_histy.set_xticks([0, 0.5, 1])
    ax_histx.set_ybound(0,1)
    ax_histy.set_xbound(0,1)
    ax_histx.tick_params(labelsize=12)
    ax_histy.tick_params(labelsize=12)
    ax_histx.set_ylabel('% of Samples', fontsize=12, weight='bold')
    ax_histy.set_xlabel('% of Samples', fontsize=12, weight='bold')
    ax_histy.set_xticklabels(labels=[0,0.5,1.0],rotation=270)
    ax_histy.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))

    ax.set_xlabel('Confidence', fontsize=18, weight='bold')
    ax.set_ylabel('Accuracy', fontsize=18, weight='bold')
    ax.tick_params(labelsize=16)
    ax.set_xbound(0, 1.0)
    ax.set_ybound(0, 1.0)

    if epoch is not None:
        plt.title(title+' Epoch: '+str(epoch), fontsize=18,
                  fontweight="bold", x=-4, y=1.37)
    # if title is not None:
    #     ax.set_title(title, fontsize=16, fontweight="bold", pad=-3)

    plt.rcParams["font.weight"] = "bold"
    ax.legend(fontsize=18)

    bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.7)
    ax.text(0.95, 0.15,
            "ACC="+str(round(acc,3)*100) +"%"+'\n'+"Avg. Conf.="+str(round(p_value.mean().tolist(),3)*100)+"%"+'\n'+"ECE="+str(round(ece,3)*100)+"%",
            ha="right", va="center", size=16,
            bbox=bbox_props)
    ax_histx.text(p_value.mean().tolist()-0.03, 0.5, "Avg.", rotation=90,
                  ha="center", va="center", size=16)
    ax_histy.text(0.5, acc-0.04, "Avg.",
                  ha="center", va="center", size=16)
    # plt.savefig(path+'/'+ title + '_epoch_' + str(epoch) +'.png', format='png', dpi=300,
    #             pad_inches=0, bbox_inches = 'tight')
    plt.show()





if __name__ == "__main__":
    output_softmax=torch.Tensor([[0.9,0.1],[0.9,0.1],[0.9,0.1],
                                 [0.67,0.23],[0.52,0.48],[0.73,0.27],
                                 [0.43,0.57],[0.44,0.56],[0.21,0.79],
                                 [0.05,0.95]])
    ground_truth=torch.Tensor([0,0,0,0,0,1,1,1,1,1])
    plot_acc_calibration('./results',output_softmax, ground_truth,0.9, 0.33111, title = 'xxx', epoch=0)
