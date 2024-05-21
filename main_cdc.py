'''
@File  :main_cdc.py
@Date  :2023/1/29 16:26
@Desc  :
'''

import logging
import re
import warnings
import wandb

logging.captureWarnings(True)
warnings.filterwarnings('always', category=DeprecationWarning,
                        module=r'^{0}.'.format(re.escape(__name__)))
warnings.warn("This is a DeprecationWarning",category=DeprecationWarning)

import argparse
import os
import torch
import random
import numpy as np

from cdc.args import parse_cfg, get_model, get_strong_transformations,\
    get_val_transformations, get_standard_transformations,\
    get_train_dataloader, get_val_dataloader,\
    get_train_dataset,get_val_dataset, get_optimizer
from cdc.utils.evaluate_utils import get_predictions, \
    hungarian_evaluate, calibration_evaluate
from cdc.methods.calibrate_train import initialize_weights, train_cali
from cdc.backbones.models import CaliMLP
FLAGS = argparse.ArgumentParser(description='CDC Model')
FLAGS.add_argument('--config_env', help='Location of path config file')
FLAGS.add_argument('--config_exp', help='Location of experiments config file')

def main():
    args = FLAGS.parse_args()
    cfg = parse_cfg(args.config_env, args.config_exp)
    print(cfg)
    # Data
    print('Get dataset and dataloaders')
    strong_transformations = get_strong_transformations(cfg)
    standard_transformations = get_standard_transformations(cfg)
    val_transformations = get_val_transformations(cfg)

    train_dataset = get_train_dataset(cfg, {'val': val_transformations,
                                            'standard': standard_transformations,
                                            'augment': strong_transformations},
                                        split=cfg['data']['split'], augmented = True)
    val_dataset = get_val_dataset(cfg, val_transformations)
    train_dataloader = get_train_dataloader(cfg, train_dataset)
    val_dataloader = get_val_dataloader(cfg, val_dataset)
    print('Strong transforms:', strong_transformations)
    print('Standard transforms:', standard_transformations)
    print('Validation transforms:', val_transformations)
    print('Train samples %d - Val samples %d' %(len(train_dataset), len(val_dataset)))

    # Model
    print('Get model')
    model = get_model(cfg, cfg['pretext']['enable'])

    cali_mlp = CaliMLP(cfg)
    cali_mlp = torch.nn.DataParallel(cali_mlp)
    cali_mlp = cali_mlp.cuda()

    # Optimizer
    print('Get optimizer')
    optimizer_clu = get_optimizer(cfg, model)
    optimizer_cali = torch.optim.Adam(cali_mlp.parameters(), lr=cfg['optimizer']['lr'],
                                      **cfg['optimizer']['kwargs'])
    # wandb
    wandb.watch(model, log="all")

    # Checkpoint
    if os.path.exists(cfg['cdc_checkpoint']):
        print('Restart from checkpoint {}'.format(cfg['cdc_checkpoint']))
        checkpoint = torch.load(cfg['cdc_checkpoint'], map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        cali_mlp.load_state_dict(checkpoint['cali_mlp'])
        optimizer_all.load_state_dict(checkpoint['optimizer_clu'])
        optimizer_cali.load_state_dict(checkpoint['optimizer_cali'])
        start_epoch = checkpoint['epoch']
    else:
        print('No checkpoint file at {}'.format(cfg['cdc_checkpoint']))
        start_epoch = 0

    # Evaluate
    predictions, features = get_predictions(cfg, val_dataloader, model, return_features=True)
    clustering_stats = hungarian_evaluate(cfg, cfg['cdc_dir'], start_epoch, 0,
                                          predictions, title=cfg['cluster_eval']['plot_title'],
                                          compute_confusion_matrix=False)
    print(clustering_stats)

    # Initialize weights
    if start_epoch == 0:
        initialize_weights(cfg, model, cali_mlp, features, val_dataloader)
    # Main loop
    print('Starting main loop', 'blue')
    best_acc = -1
    for epoch in range(start_epoch, cfg['max_epochs']):
        print('Epoch %d/%d' % (epoch + 1, cfg['max_epochs']))
        # Train
        print('Train ...')
        train_cali(cfg, train_dataloader, cali_mlp, model, optimizer_cali, optimizer_clu, epoch, start_epoch)
        # Evaluate
        if (epoch+1) % 1 == 0:
            print('Make prediction on validation set ...')
            predictions = get_predictions(cfg, val_dataloader, model)
            clustering_stats = hungarian_evaluate(cfg, cfg['cdc_dir'], epoch, 0, predictions,
                                                  title=cfg['cluster_eval']['plot_title'],
                                                  compute_confusion_matrix=False)
            print('CDC-Clu ', clustering_stats)
            predictions = get_predictions(cfg, val_dataloader, model, cali_mlp = cali_mlp)
            clustering_stats = calibration_evaluate(cfg, cfg['cdc_dir'], epoch, 0, predictions,
                                                  title=cfg['cluster_eval']['plot_title'],
                                                  compute_confusion_matrix=False)
            print('CDC-Cal ', clustering_stats)
        # Checkpoint
        print('Checkpoint ...')
        torch.save({'optimizer_clu': optimizer_all.state_dict(),
                    'optimizer_cali': optimizer_cali.state_dict(),
                    'model': model.state_dict(),
                    'cali_mlp': cali_mlp.state_dict(),
                    'epoch': epoch + 1},
                   cfg['cdc_checkpoint'])
        if best_acc<clustering_stats['ACC']:
            torch.save({
                        'model': model.state_dict(),
                        'cali_mlp': cali_mlp.state_dict(),
                        'epoch': epoch + 1},
                       cfg['cdc_best_model'])
            best_acc = clustering_stats['ACC']


if __name__ == "__main__":
    seed = 1024
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_grad_enabled(True)
    main()
    wandb.finish()