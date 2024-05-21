"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from sklearn import metrics
from scipy.optimize import linear_sum_assignment
from torchmetrics.functional import calibration_error
from scipy.stats import gaussian_kde
from scipy import interpolate
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as mtick

def get_feature_dimensions_backbone(cfg):
    if cfg['backbone']['name'] == 'resnet18':
        return 512
    elif cfg['backbone']['name'] == 'resnet34':
        return 512
    elif cfg['backbone']['name'] == 'resnet50':
        return 2048
    else:
        raise NotImplementedError

@torch.no_grad()
def get_predictions(cfg, dataloader, model, return_features=False, is_train=False, cali_mlp = None):
    # Make predictions on a dataset with neighbors
    # print("checkpoint 1")
    model.eval()
    if cali_mlp is not None:
        cali_mlp.eval()
    predictions = [[] for _ in range(cfg['backbone']['nheads'])]
    probs = [[] for _ in range(cfg['backbone']['nheads'])]
    targets = []
    if return_features:
        ft_dim = cfg['backbone']['feat_dim']
        features = torch.zeros((len(dataloader.sampler), ft_dim)).cuda()
    if is_train:
        output_val = []
    ptr = 0
    # print("checkpoint 1.1")
    with torch.no_grad():
        for batch in dataloader:
            if is_train:
                images = batch['image'].cuda(non_blocking=True)
                targets_ = batch['target'].cuda(non_blocking=True)
            else:
                # print("checkpoint 1.2")
                images, targets_ = batch
            bs = images.shape[0]
            if cfg['method']=='cc' or 'cc' in cfg['name']:
                output = model(images.cuda(non_blocking=True),
                        forward_pass='test')
            elif cali_mlp is not None:
                fea = model(images.cuda(non_blocking=True),
                                     forward_pass='backbone')
                output = [cali_mlp(fea, forward_pass='calibration')]
            else:
                # print("checkpoint 1.3")
                res = model(images.cuda(non_blocking=True),
                        forward_pass='return_all')
                output = res['output']
            if return_features:
                # print("checkpoint 1.4")
                features[ptr: ptr+bs] = res['features']
                ptr += bs
            for i, output_i in enumerate(output):
                # print("checkpoint 1.5")
                predictions[i].append(torch.argmax(output_i, dim=1))
                if cfg['method'] == 'cc' or 'cc' in cfg['name']:
                    probs[i].append(output_i)
                else:
                    probs[i].append(F.softmax(output_i, dim=1))
            targets.append(targets_)
            if is_train:
                output_val.append(output[0])
    # print("checkpoint 2")
    predictions = [torch.cat(pred_, dim = 0).cpu() for pred_ in predictions]
    probs = [torch.cat(prob_, dim=0).cpu() for prob_ in probs]
    targets = torch.cat(targets, dim=0)
    out = [{'predictions': pred_, 'probabilities': prob_, 'targets': targets} for pred_, prob_ in zip(predictions, probs)]
    if return_features:
        return out, features
    elif is_train:
        output_val = torch.cat(output_val, dim=0)
        return out, output_val
    else:
        return out

@torch.no_grad()
def _hungarian_match(flat_preds, flat_targets, preds_k, targets_k):
    # Based on implementation from IIC
    num_samples = flat_targets.shape[0]

    # assert (preds_k == targets_k)  # one to one
    num_k = preds_k
    num_correct = np.zeros((num_k, num_k))

    for c1 in range(num_k):
        for c2 in range(num_k):
            # elementwise, so each sample contributes once
            votes = int(((flat_preds == c1) * (flat_targets == c2)).sum())
            num_correct[c1, c2] = votes

    # num_correct is small
    match = linear_sum_assignment(num_samples - num_correct)
    match = np.array(list(zip(*match)))

    # return as list of tuples, out_c to gt_c
    res = []
    for out_c, gt_c in match:
        res.append((out_c, gt_c))

    return res

def confusion_matrix(predictions, gt, class_names, output_file=None):
    # Plot confusion_matrix and store result to output_file
    import sklearn.metrics
    import matplotlib.pyplot as plt
    confusion_matrix = sklearn.metrics.confusion_matrix(gt, predictions)
    confusion_matrix = confusion_matrix / np.sum(confusion_matrix, 1)

    fig, axes = plt.subplots(1)
    plt.imshow(confusion_matrix, cmap='Blues')
    axes.set_xticks([i for i in range(len(class_names))])
    axes.set_yticks([i for i in range(len(class_names))])
    axes.set_xticklabels(class_names, ha='right', fontsize=8, rotation=40)
    axes.set_yticklabels(class_names, ha='right', fontsize=8)

    for (i, j), z in np.ndenumerate(confusion_matrix):
        if i == j:
            axes.text(j, i, '%d' % (100 * z), ha='center', va='center', color='white', fontsize=6)
        else:
            pass

    plt.tight_layout()
    if output_file is None:
        plt.show()
    else:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def plot_acc_calibration(path, output_softmax, ground_truth, acc, ece,
                         n_bins = 15, title = None, epoch=None):
    p_value = np.max(output_softmax, 1)
    pred_label = np.argmax(output_softmax, 1)
    confidence_all, confidence_acc = np.zeros(n_bins), np.zeros(n_bins)

    sub_n_bins = n_bins * 3
    bins = np.arange(0, 1.0 + 1 / sub_n_bins-0.0001, 1 / sub_n_bins)
    sub_weights = np.ones(len(ground_truth)) / float(len(ground_truth))
    sub_acc = np.zeros_like(ground_truth, dtype=float)
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
        sub_acc[index] = confidence_acc[interval]

    start = np.around(1/n_bins/2, 3)
    step = np.around(1/n_bins, 3)
    # plt.figure(figsize=(6, 5))
    plt.rcParams["font.weight"] = "bold"
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.bar(np.around(np.arange(start, 1.0, step), 3),
            np.around(np.arange(start, 1.0, step), 3), alpha=0.7, width=0.05, color='orange', label='Expected')
    ax.bar(np.around(np.arange(start, 1.0, step), 3), confidence_acc,
           alpha=0.9, width=0.05, color='dodgerblue', label='Outputs')



    # ax.set_aspect(1.)
    ax.plot([0,1], [0,1], ls='--',c='k')
    divider = make_axes_locatable(ax)
    ax_histx = divider.append_axes("top", 0.6, pad=0.3, sharex=ax)
    ax_histy = divider.append_axes("right", 0.6, pad=0.3, sharey=ax)
    ax_histx.xaxis.set_tick_params(labelbottom=False)
    ax_histy.yaxis.set_tick_params(labelleft=False)

    # ax_histx.hist(p_value, bins=bins, edgecolor='white',
    #               color='lightblue', weights=sub_weights)
    # ax_histy.hist(sub_acc, bins=bins, orientation='horizontal',
    #               edgecolor='white', color='lightblue', weights=sub_weights)

    density = gaussian_kde(p_value)
    xs = np.linspace(0, 1, 30)
    density.covariance_factor = lambda: .05
    density._compute_covariance()
    f = interpolate.interp1d(xs, density(xs)/density(xs).sum(), kind='cubic')
    nx = np.linspace(0, 1, 100)
    ny = f(nx)
    ax_histx.plot(nx, ny,
                  c='dodgerblue', linewidth=3)
    density = gaussian_kde(sub_acc)
    xs = np.linspace(0, 1, 30)
    density.covariance_factor = lambda: .05
    density._compute_covariance()
    f = interpolate.interp1d(xs, density(xs)/density(xs).sum(), kind='cubic')
    nx = np.linspace(0, 1, 100)
    ny = f(nx)
    ax_histy.plot(ny, nx,
                  c='dodgerblue', linewidth=3)

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


    ax.legend(fontsize=18)

    bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.7)
    ax.text(0.95, 0.15,
            "ACC="+str(round(acc*100,1)) +"%"+'\n'+"Avg. Conf.="+str(round(p_value.mean()*100,1))+"%"+'\n'+"ECE="+str(round(ece*100,1))+"%",
            ha="right", va="center", size=16,
            bbox=bbox_props)
    ax_histx.text(p_value.mean().tolist()-0.03, 0.5, "Avg.", rotation=90,
                  ha="center", va="center", size=16)
    ax_histy.text(0.5, acc-0.04, "Avg.",
                  ha="center", va="center", size=16)
    plt.savefig(path+'/'+ title + '_epoch_' + str(epoch) +'.png', format='png', dpi=300,
                pad_inches=0, bbox_inches = 'tight')
    # plt.show()
    plt.close()

@torch.no_grad()
def hungarian_evaluate(cfg: object, path: object, epoch: object, subhead_index: object, all_predictions: object,
                       title: object, class_names: object = None,
                       compute_confusion_matrix: object = True, confusion_matrix_file: object = None) -> object:
    # Evaluate model based on hungarian matching between predicted cluster assignment and gt classes.
    # This is computed only for the passed subhead index.
    # Hungarian matching
    head = all_predictions[subhead_index]
    targets = head['targets'].cuda()
    predictions = head['predictions'].cuda()
    probs = head['probabilities'].cuda()
    num_classes = torch.unique(targets).numel()
    num_elems = targets.size(0)

    match = _hungarian_match(predictions, targets, preds_k=probs.shape[1], targets_k=num_classes)
    reordered_preds = torch.zeros(num_elems, dtype=predictions.dtype).cuda()
    reordered_probs = torch.zeros((num_elems,num_classes), dtype=probs.dtype).cuda()
    for pred_i, target_i in match:
        reordered_preds[predictions == int(pred_i)] = int(target_i)
        reordered_probs[:,target_i]=probs[:,pred_i]
    # Gather performance metrics
    acc = int((reordered_preds == targets).sum()) / float(num_elems)
    nmi = metrics.normalized_mutual_info_score(targets.cpu().numpy(), predictions.cpu().numpy())
    ari = metrics.adjusted_rand_score(targets.cpu().numpy(), predictions.cpu().numpy())

    _, preds_top5 = probs.topk(5, 1, largest=True)
    reordered_preds_top5 = torch.zeros_like(preds_top5)
    for pred_i, target_i in match:
        reordered_preds_top5[preds_top5 == int(pred_i)] = int(target_i)
    correct_top5_binary = reordered_preds_top5.eq(targets.view(-1,1).expand_as(reordered_preds_top5))
    top5 = float(correct_top5_binary.sum()) / float(num_elems)
    ece = calibration_error(reordered_probs, targets, n_bins=15).item()
    conf = reordered_probs.max(1)[0].mean().item()

    dist = reordered_preds.unique(return_counts=True)[1]
    imb_ratio = (dist.max()/dist.min()).item()

    # Compute confusion matrix
    if compute_confusion_matrix:
        confusion_matrix(reordered_preds.cpu().numpy(), targets.cpu().numpy(), 
                            class_names, confusion_matrix_file)
    # print("checkpoint 4")
    # cal calibration
    if (epoch == 0) or (epoch+1) % cfg['cluster_eval']['plot_freq'] == 0:
        try:
            plot_acc_calibration(path, reordered_probs.cpu().numpy(),
                             targets.cpu().numpy(), acc, ece,
                             n_bins = 15, title= title,
                             epoch=epoch+1)
            # print("checkpoint 5")
        except:
            pass
    if epoch ==998:
        wandb.log({
            'TACC': round(acc*100,6),'TNMI': round(nmi*100,6),
            'TARI': round(ari*100,6), 'TECE': round(ece*100, 6), 'CONF': round(conf*100,6),
            'IMBRATO':round(imb_ratio,6),
            'TACC Top-5': round(top5*100,6),
        })
    else:
        wandb.log({
            'epoch': epoch,
            'ACC': round(acc*100, 6), 'NMI': round(nmi*100, 6),
            'ARI': round(ari*100, 6), 'ECE': round(ece*100, 6), 'CONF': round(conf*100,6),
            'IMBRATO': round(imb_ratio, 6),
            'ACC Top-5': round(top5*100, 6),
        })

    return {'ACC': round(acc*100,6),'NMI': round(nmi*100,6),
            'ARI': round(ari*100,6), 'ECE': round(ece*100,6), 'CONF': round(conf*100,6),
            'IMBRATO': round(imb_ratio, 6),
            'ACC Top-5': round(top5*100,6),'hungarian_match': match}


@torch.no_grad()
def calibration_evaluate(cfg: object, path: object, epoch: object, subhead_index: object, all_predictions: object,
                       title: object, class_names: object = None,
                       compute_confusion_matrix: object = True, confusion_matrix_file: object = None) -> object:
    # Evaluate model based on hungarian matching between predicted cluster assignment and gt classes.
    # This is computed only for the passed subhead index.

    # Hungarian matching
    head = all_predictions[subhead_index]
    targets = head['targets'].cuda()
    predictions = head['predictions'].cuda()
    probs = head['probabilities'].cuda()
    num_classes = torch.unique(targets).numel()
    num_elems = targets.size(0)

    match = _hungarian_match(predictions, targets, preds_k=num_classes, targets_k=num_classes)
    reordered_preds = torch.zeros(num_elems, dtype=predictions.dtype).cuda()
    reordered_probs = torch.zeros((num_elems, num_classes), dtype=probs.dtype).cuda()
    for pred_i, target_i in match:
        reordered_preds[predictions == int(pred_i)] = int(target_i)
        reordered_probs[:, target_i] = probs[:, pred_i]
    # Gather performance metrics
    acc = int((reordered_preds == targets).sum()) / float(num_elems)
    nmi = metrics.normalized_mutual_info_score(targets.cpu().numpy(), predictions.cpu().numpy())
    ari = metrics.adjusted_rand_score(targets.cpu().numpy(), predictions.cpu().numpy())

    _, preds_top5 = probs.topk(5, 1, largest=True)
    reordered_preds_top5 = torch.zeros_like(preds_top5)
    for pred_i, target_i in match:
        reordered_preds_top5[preds_top5 == int(pred_i)] = int(target_i)
    correct_top5_binary = reordered_preds_top5.eq(targets.view(-1, 1).expand_as(reordered_preds_top5))
    top5 = float(correct_top5_binary.sum()) / float(num_elems)
    ece = calibration_error(reordered_probs, targets, n_bins=15).item()
    conf = reordered_probs.max(1)[0].mean().item()
    # Compute confusion matrix
    if compute_confusion_matrix:
        confusion_matrix(reordered_preds.cpu().numpy(), targets.cpu().numpy(),
                         class_names, confusion_matrix_file)

    # cal calibration
    if (epoch == 0) or (epoch + 1) % cfg['cluster_eval']['plot_freq'] == 0:
        try:
            plot_acc_calibration(path, reordered_probs.cpu().numpy(),
                                 targets.cpu().numpy(), acc, ece,
                                 n_bins=15, title=title+'cali',
                                 epoch=epoch + 1)
        except:
            pass

    wandb.log({
        'epoch_cali': epoch,
        'ACC_cali': round(acc*100, 6), 'NMI_cali': round(nmi*100, 6),
        'ARI_cali': round(ari*100, 6), 'ECE_cali': round(ece*100, 6), 'CONF_cali': round(conf*100, 6),
        'ACC Top-5_cali': round(top5*100, 6),
    })

    return {'ACC': round(acc*100, 6), 'NMI': round(nmi*100, 6),
            'ARI': round(ari*100, 6), 'ECE': round(ece*100, 6), 'CONF': round(conf*100, 6),
            'ACC Top-5': round(top5*100, 6), 'hungarian_match': match}
