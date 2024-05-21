'''
@File  :cluster_metrics.py
@Date  :2023/1/20 21:47
@Desc  :
'''
from typing import Tuple
import torch
import torch.nn.functional as F
import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from torchmetrics.functional import calibration_error
from torchmetrics.metric import Metric
from pretrain.utils.torch_clustering import PyTorchKMeans, evaluate_clustering


class ClusterMetrics(Metric):
    def __init__(
        self,
        preds_k: int = 10,
        targets_k: int = 10,
        random_state: int = 10,
        ece_bins: int = 15,
        ece_norm: str = "l1",
    ):
        super().__init__(compute_on_step=False)
        self.preds_k = preds_k
        self.targets_k = targets_k
        self.random_state = random_state
        self.ece_bins = ece_bins
        self.ece_norm = ece_norm

        self.add_state("test_features", default=[], persistent=False)
        self.add_state("z_features", default=[], persistent=False)
        self.add_state("test_outputs", default=[], persistent=False)
        self.add_state("test_targets", default=[], persistent=False)

    def update(
        self,
        test_features: torch.Tensor = None,
        z_features: torch.Tensor = None,
        test_outputs: torch.Tensor = None,
        test_targets: torch.Tensor = None,
    ):
        """Updates the memory banks. If  test features are passed as input, the
        corresponding test targets must be passed as well.

        Args:
            test_features (torch.Tensor, optional): a batch of test features. Defaults to None.
            test_targets (torch.Tensor, optional): a batch of test targets. Defaults to None.
        """
        assert (test_features is None) == (z_features is None)
        assert (z_features is None) == (test_outputs is None)
        assert (test_outputs is None) == (test_targets is None)

        if test_features is not None:
            assert test_features.size(0) == z_features.size(0)
            assert z_features.size(0) == test_outputs.size(0)
            assert test_outputs.size(0) == test_targets.size(0)
            self.test_features.append(test_features.detach())
            self.z_features.append(z_features.detach())
            self.test_outputs.append(test_outputs.detach())
            self.test_targets.append(test_targets.detach())


    @torch.no_grad()
    def compute(self) -> Tuple[float]:
        test_features = torch.cat(self.test_features)
        z_features = torch.cat(self.z_features)
        test_outputs = torch.cat(self.test_outputs)
        test_targets = torch.cat(self.test_targets)

        test_features_numpy = test_features.cpu().numpy()
        z_features_numpy = z_features.cpu().numpy()
        test_outputs_argmax = test_outputs.max(1)[1].cpu().numpy()
        gt = test_targets.cpu().numpy()

        clustermd = KMeans(n_clusters=self.preds_k, random_state=self.random_state)
        clustermd.fit(test_features_numpy)
        plabels = clustermd.labels_
        f_match = _hungarian_match(plabels, gt, preds_k=self.preds_k, targets_k=self.targets_k)

        reordered_preds = np.zeros(gt.shape[0])
        for pred_i, target_i in f_match:
            reordered_preds[plabels == int(pred_i)] = int(target_i)

        #
        f_acc = int((reordered_preds == gt).sum()) / float(gt.shape[0])
        f_nmi = metrics.normalized_mutual_info_score(gt, plabels)
        f_ari = metrics.adjusted_rand_score(gt, plabels)
        f_micro_p = metrics.precision_score(gt, reordered_preds, average='micro')
        f_macro_p = metrics.precision_score(gt, reordered_preds, average='macro')
        f_micro_r = metrics.recall_score(gt, reordered_preds, average='micro')
        f_macro_r = metrics.recall_score(gt, reordered_preds, average='macro')
        f_micro_f1 = metrics.f1_score(gt, reordered_preds, average='micro')
        f_macro_f1 = metrics.f1_score(gt, reordered_preds, average='macro')
        f_purity = purity_score(gt, plabels)


        clustermd = KMeans(n_clusters=self.preds_k, random_state=self.random_state)
        clustermd.fit(z_features_numpy)
        plabels = clustermd.labels_
        z_match = _hungarian_match(plabels, gt, preds_k=self.preds_k, targets_k=self.targets_k)
        reordered_preds = np.zeros(gt.shape[0])
        for pred_i, target_i in z_match:
            reordered_preds[plabels == int(pred_i)] = int(target_i)
        # after projector
        z_acc = int((reordered_preds == gt).sum()) / float(gt.shape[0])
        z_nmi = metrics.normalized_mutual_info_score(gt, plabels)
        z_ari = metrics.adjusted_rand_score(gt, plabels)
        z_micro_p = metrics.precision_score(gt, reordered_preds, average='micro')
        z_macro_p = metrics.precision_score(gt, reordered_preds, average='macro')
        z_micro_r = metrics.recall_score(gt, reordered_preds, average='micro')
        z_macro_r = metrics.recall_score(gt, reordered_preds, average='macro')
        z_micro_f1 = metrics.f1_score(gt, reordered_preds, average='micro')
        z_macro_f1 = metrics.f1_score(gt, reordered_preds, average='macro')
        z_purity = purity_score(gt, plabels)

        linear_match = _hungarian_match(test_outputs_argmax, gt, preds_k=self.preds_k, targets_k=self.targets_k)
        linear_acc = int((test_outputs_argmax == gt).sum()) / float(gt.shape[0])
        linear_nmi = metrics.normalized_mutual_info_score(gt, test_outputs_argmax)
        linear_ari = metrics.adjusted_rand_score(gt, test_outputs_argmax)
        linear_ece = calibration_error(test_outputs, test_targets,
                                n_bins=self.ece_bins, norm=self.ece_norm)

        reordered_preds = np.zeros(gt.shape[0])
        for pred_i, target_i in linear_match:
            reordered_preds[plabels == int(pred_i)] = int(target_i)
        linear_micro_p = metrics.precision_score(gt, reordered_preds, average='micro')
        linear_macro_p = metrics.precision_score(gt, reordered_preds, average='macro')
        linear_micro_r = metrics.recall_score(gt, reordered_preds, average='micro')
        linear_macro_r = metrics.recall_score(gt, reordered_preds, average='macro')
        linear_micro_f1 = metrics.f1_score(gt, reordered_preds, average='micro')
        linear_macro_f1 = metrics.f1_score(gt, reordered_preds, average='macro')

        kwargs = {
            'metric': 'cosine',
            'distributed': False,
            'random_state': 10,
            'n_clusters': test_outputs.shape[1],
            'verbose': False
        }
        clustering_model = PyTorchKMeans(init='k-means++', max_iter=300, tol=1e-4, **kwargs)
        try:
            psedo_labels = clustering_model.fit_predict(test_features)
            results1 = evaluate_clustering(gt, psedo_labels.cpu().numpy(), eval_metric=['acc','nmi','ari'],phase='test')
        except:
            psedo_labels = clustering_model.fit_predict(F.normalize(test_features))
            results1 = evaluate_clustering(gt, psedo_labels.cpu().numpy(), eval_metric=['acc','nmi','ari'],phase='test')

        kwargs = {
            'metric': 'cosine',
            'distributed': False,
            'random_state': 10,
            'n_clusters': test_outputs.shape[1],
            'verbose': False
        }
        clustering_model = PyTorchKMeans(init='k-means++', max_iter=300, tol=1e-4, **kwargs)
        try:
            psedo_labels = clustering_model.fit_predict(z_features)
            results2 = evaluate_clustering(gt, psedo_labels.cpu().numpy(), eval_metric=['acc', 'nmi', 'ari'],
                                           phase='test')
        except:
            psedo_labels = clustering_model.fit_predict(F.normalize(test_features))
            results2 = evaluate_clustering(gt, psedo_labels.cpu().numpy(), eval_metric=['acc', 'nmi', 'ari'],
                                           phase='test')

        print()
        print("Cluster Evaluation Before Projector(Kmeans++)",results1)
        print("Cluster Evaluation After Projector(Kmeans++)",results2)



        return f_acc, f_nmi, f_ari, f_match, z_acc, z_nmi, z_ari, z_match, \
               linear_acc, linear_nmi, linear_ari, linear_ece, linear_match, \
               results1['test_acc'], results1['test_nmi'], results1['test_ari'], \
               results2['test_acc'], results2['test_nmi'], results2['test_ari'], \
               f_micro_p, f_macro_p, f_micro_r, f_macro_r, f_micro_f1, f_macro_f1, f_purity, \
               z_micro_p, z_macro_p, z_micro_r, z_macro_r, z_micro_f1, z_macro_f1, z_purity, \
               linear_micro_p, linear_macro_p, linear_micro_r, \
               linear_macro_r, linear_micro_f1, linear_macro_f1



    def delete(
            self,
    ):
        self.test_features = []
        self.z_features = []
        self.test_outputs = []
        self.test_targets = []


@torch.no_grad()
def _hungarian_match(flat_preds, flat_targets, preds_k, targets_k):
        # Based on implementation from IIC
        num_samples = flat_targets.shape[0]

        assert (preds_k == targets_k)  # one to one
        num_k = preds_k
        num_correct = np.zeros((num_k, num_k)).astype(float)


        for c1 in range(num_k):
            for c2 in range(num_k):
                # elementwise, so each sample contributes once
                votes = int(((flat_preds == c1) * (flat_targets == c2)).sum())
                num_correct[c1, c2] = votes

        # num_correct is small
        match = linear_sum_assignment(num_samples - num_correct)
        match = np.array(list(zip(*match))).astype(float)

        # return as list of tuples, out_c to gt_c
        res = []
        for out_c, gt_c in match:
            res.append((out_c, gt_c))

        return res
def purity_score(y_true, y_pred):
    """Purity score
        Args:
            y_true(np.ndarray): n*1 matrix Ground truth labels
            y_pred(np.ndarray): n*1 matrix Predicted clusters

        Returns:
            float: Purity score
    """
    # matrix which will hold the majority-voted labels
    y_voted_labels = np.zeros(y_true.shape)
    # Ordering labels
    ## Labels might be missing e.g with set like 0,2 where 1 is missing
    ## First find the unique labels, then map the labels to an ordered set
    ## 0,2 should become 0,1
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true==labels[k]] = ordered_labels[k]
    # Update unique labels
    labels = np.unique(y_true)
    # We set the number of bins to be n_classes+2 so that
    # we count the actual occurence of classes between two consecutive bins
    # the bigger being excluded [bin_i, bin_i+1[
    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred==cluster], bins=bins)
        # Find the most present label in the cluster
        winner = np.argmax(hist)
        y_voted_labels[y_pred==cluster] = winner

    return metrics.accuracy_score(y_true, y_voted_labels)