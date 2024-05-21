import torch
import torch.nn.functional as F
import wandb
from cdc.utils.torch_clustering import PyTorchKMeans
from cdc.utils.evaluate_utils import get_predictions, hungarian_evaluate

def orth_train(W, n_samples, scale = 5, epochs=2000, use_relu = False):
    Z = W.clone().cuda()
    Z.requires_grad = True
    W_ = W.clone().cuda()
    W_.requires_grad = True
    labels = torch.arange(0, n_samples).cuda()
    optimizer = torch.optim.SGD([Z, W_], lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)
    criterion = torch.nn.CrossEntropyLoss()
    # with torch.enable_grad():
    for i in range(epochs):
        if use_relu:
            z = F.relu(Z)
        else:
            z = Z
        w = W_
        L2_z = F.normalize(z, dim=1)
        L2_w = F.normalize(w, dim=1)
        out = F.linear(L2_z, L2_w)
        loss = criterion(out * scale, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
    return W_.detach()
def initialize_weights(cfg, model, cali_mlp, features, val_dataloader):
    features_zscore = (features - features.mean(1).reshape(-1, 1)) / features.std(1).reshape(-1, 1)
    features_zscore = F.normalize(features_zscore, dim=1)

    KMeans_512 = PyTorchKMeans(init='k-means++', n_clusters=512, verbose=False)
    proto_label = KMeans_512.fit_predict(features_zscore)
    W1 = KMeans_512.cluster_centers_

    #linear(512,512)
    H = torch.mm(features, W1.T)
    #BN
    # H = (H - H.mean(0)) / H.std(0)
    H = model.module.cluster_head[0][1](H).detach().clone()
    #relu
    # H = torch.nn.functional.relu(H)
    H = model.module.cluster_head[0][2](H).detach().clone()

    H_zscore = (H - H.mean(1).reshape(-1, 1)) / H.std(1).reshape(-1, 1)
    H_zscore = F.normalize(H_zscore, dim=1)

    KMeans_c = PyTorchKMeans(init='k-means++', n_clusters=cfg['backbone']['nclusters'], verbose=False)
    class_label = KMeans_c.fit_predict(H_zscore)
    W2 = KMeans_c.cluster_centers_

    W1_modi = orth_train(W1, 512, use_relu=True)
    W2_modi = orth_train(W2, cfg['backbone']['nclusters'], use_relu=True)

    O = torch.mm(torch.mm(features, W1.T), W2.T)
    print(F.softmax(O, dim=1).max(1)[1].unique(return_counts=True))
    O = torch.mm(torch.mm(features, W1_modi.T) , W2_modi.T)
    print(F.softmax(O, dim=1).max(1)[1].unique(return_counts=True))

    torch.nn.init.zeros_(model.module.cluster_head[0][0].bias)
    torch.nn.init.zeros_(model.module.cluster_head[0][3].bias)
    model.module.cluster_head[0][0].weight.data = W1_modi.clone()
    model.module.cluster_head[0][3].weight.data = W2_modi.clone()

    torch.nn.init.zeros_(cali_mlp.module.calibration_head[0].bias)
    torch.nn.init.zeros_(cali_mlp.module.calibration_head[3].bias)
    cali_mlp.module.calibration_head[0].weight.data = W1_modi.clone()
    cali_mlp.module.calibration_head[3].weight.data = W2_modi.clone()

    predictions = get_predictions(cfg, val_dataloader, model)
    clustering_stats = hungarian_evaluate(cfg, cfg['cdc_checkpoint'], 0, 0,
                                          predictions, title=cfg['cluster_eval']['plot_title'],
                                          compute_confusion_matrix=False)
    print(clustering_stats)
def train_cali(cfg, train_dataloader, cali_mlp, model, optimizer_cali, optimizer_all, epoch, start_epoch):
    loss_clu, loss_cali = [],[]
    loss_ces,loss_ens,loss_coss = [],[],[]
    for step, batch in enumerate(train_dataloader):
        model.zero_grad()
        optimizer_all.zero_grad()
        import time
        st = time.time()
        images = batch['image'].cuda(non_blocking=True)
        images_augmented = batch['image_augmented'].cuda(non_blocking=True)
        images_val = batch['val'].cuda(non_blocking=True)
        images_index = batch['index'].cuda(non_blocking=True)
        gt = batch['target'].cuda(non_blocking=True)

        model.train()
        cali_mlp.train()
        with torch.no_grad():
            feature_val = model(images_val, forward_pass='backbone')
            output_clu_val = model(feature_val, forward_pass='head')[0]

            feature_weak = model(images, forward_pass='backbone')
            output_clu = model(feature_weak, forward_pass='head')[0]
            output_cali = cali_mlp(feature_weak, forward_pass='calibration')
        feature_norm1 = F.normalize(feature_val, p=1, dim=1)

        clu_softmax = F.softmax(output_clu, dim=1)
        cali_softmax = F.softmax(output_cali, dim=1)
        clu_prob, clu_label = torch.max(clu_softmax, dim=1)
        cali_prob, cali_label = torch.max(cali_softmax, dim=1)

        proto_pseudo = cali_label
        selected_num = cfg['method_kwargs']['per_class_selected_num']
        # selected_num = int(output_cali.shape[0] / output_cali.shape[1])
        selected_idx = torch.zeros(len(cali_softmax)).cuda()
        for label_idx in range(output_clu.shape[1]):
            per_label_mask = cali_softmax[:, label_idx].sort(descending=True)[1][:selected_num]
            sel = int(cali_prob[per_label_mask].mean() * selected_num)
            selected_idx[per_label_mask[:sel]]=1
        selected_idx = selected_idx==1

        cluster_num = cfg['method_kwargs']['super_cluster_num']
        KMeans_all = PyTorchKMeans(init='k-means++', n_clusters=cluster_num, verbose=False)
        split_all = KMeans_all.fit_predict(feature_norm1)
        target_dict = torch.stack([F.softmax(output_clu_val, dim=1)[split_all == i].mean(0) for i in range(cluster_num)])
        super_target = target_dict[split_all]

        sub_steps = int(cfg['optimizer']['batch_size']/cfg['optimizer']['sub_batch_size'])
        sub_idxs = torch.range(0, sub_steps*cfg['optimizer']['sub_batch_size']-1).to(torch.int64).reshape(sub_steps,-1)
        for sub_step in range(sub_steps):
            sub_idx = sub_idxs[sub_step]
            output_aug = model(images_augmented[sub_idx])[0]
            sub_proto_pseudo, sub_selected_idx = proto_pseudo[sub_idx], selected_idx[sub_idx]
            loss_ce = F.cross_entropy(output_aug[sub_selected_idx], sub_proto_pseudo[sub_selected_idx])
            loss = loss_ce
            loss_ces.append(loss_ce.detach())
            loss_clu.append(loss.detach())

            optimizer_all.zero_grad()
            loss.backward()
            optimizer_all.step()

            output_cali = cali_mlp(feature_val[sub_idx], forward_pass='calibration')
            cali_prob, _ = F.softmax(output_cali, dim=1).max(1)

            loss_cos = (-super_target[sub_idx]*F.log_softmax(output_cali)).sum(1).mean()
            x_ = torch.mean(F.softmax(output_cali, dim=1), 0)
            loss_entropy = torch.sum(x_ * torch.log(x_))

            loss = loss_cos+cfg['method_kwargs']['w_en']*loss_entropy

            loss_cali.append(loss.detach())
            loss_coss.append(loss_cos.detach())
            loss_ens.append(loss_entropy.detach())

            optimizer_cali.zero_grad()
            loss.backward()
            optimizer_cali.step()
    wandb.log({
        "loss_clu":torch.stack(loss_clu).mean(),
        "loss_cali":torch.stack(loss_cali).mean(),
        "loss_ces":torch.stack(loss_ces).mean(),
        "loss_cos":torch.stack(loss_coss).mean(),
        "loss_ens":torch.stack(loss_ens).mean(),
    })
