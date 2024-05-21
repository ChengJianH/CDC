'''
@File  :args.py
@Date  :2023/1/29 16:15
@Desc  :
'''
import os

import wandb
import yaml
from easydict import EasyDict
import errno
import torch
import torchvision.transforms as transforms
from cdc.data.augment import Augment, Cutout
from cdc.data.collate import collate_custom


def mkdir_if_missing(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

def parse_cfg(config_file_env, config_file_exp):
    # Config for environment path
    with open(config_file_env, 'r') as stream:
        root_dir = yaml.safe_load(stream)['root_dir']

    with open(config_file_exp, 'r') as stream:
        config = yaml.safe_load(stream)

    cfg = EasyDict()

    # Copy
    for k, v in config.items():
        cfg[k] = v

    wandb.init(project=cfg['project'], name=cfg['name'])

    # Set paths for pretext task (These directories are needed in every stage)
    base_dir = os.path.join(root_dir, cfg['data']['dataset'])
    mkdir_if_missing(base_dir)

    if cfg['pretext']['enable']:
        pretext_dir = cfg['pretext']['dir']
        cfg['pretext_dir'] = pretext_dir
        cfg['pretext_model'] = os.path.join(pretext_dir, cfg['pretext']['ckpt'])
    if cfg['method'] == 'cdcv2':
        cdc_dir = os.path.join(base_dir, cfg['name'])
        mkdir_if_missing(cdc_dir)
        cfg['cdc_dir'] = cdc_dir
        cfg['cdc_checkpoint'] = os.path.join(cdc_dir, 'checkpoint.pth.tar')
        cfg['cdc_best_model'] = os.path.join(cdc_dir, 'best_model.pth.tar')
        cfg['cdc_model'] = os.path.join(cdc_dir, 'model.pth.tar')
    return cfg

def get_strong_transformations(cfg):
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(cfg['augmentation_strong']['crop_size']),
        Augment(cfg['augmentation_strong']['num_strong_augs']),
        transforms.ToTensor(),
        transforms.Normalize(**cfg['augmentation_strong']['normalize']),
        Cutout(
            n_holes=cfg['augmentation_strong']['cutout_kwargs']['n_holes'],
            length=cfg['augmentation_strong']['cutout_kwargs']['length'],
            random=cfg['augmentation_strong']['cutout_kwargs']['random'])])
def get_standard_transformations(cfg):
    return transforms.Compose([
        transforms.RandomResizedCrop(**cfg['augmentation_stantard']['random_resized_crop']),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(**cfg['augmentation_stantard']['normalize'])
    ])

def get_val_transformations(cfg):
    if cfg['data']['dataset'] in ["imagenetdogs", "imagenet10"]:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(cfg['augmentation_val']['crop_size']),
            transforms.ToTensor(),
            transforms.Normalize(**cfg['augmentation_val']['normalize'])])
    else:
        return transforms.Compose([
            transforms.CenterCrop(cfg['augmentation_val']['crop_size']),
            transforms.ToTensor(),
            transforms.Normalize(**cfg['augmentation_val']['normalize'])])

def get_train_dataset(cfg, transform, augmented = False, split=None):
    # Base dataset
    if cfg['data']['dataset'] == 'cifar10':
        from cdc.data.cifar20_dataset import CIFAR10
        dataset = CIFAR10(cfg['data']['train_path'],
                          train=split, transform=transform, download=True)

    elif cfg['data']['dataset'] == 'cifar20':
        from cdc.data.cifar20_dataset import CIFAR20
        dataset = CIFAR20(cfg['data']['train_path'],
                          train=split, transform=transform, download=True)

    elif cfg['data']['dataset'] == 'stl10':
        from cdc.data.stl10_dataset import STL10
        dataset = STL10(cfg['data']['train_path'],
                        split=split, transform=transform, download=True)

    elif cfg['data']['dataset'] in ["imagenet", "imagenet100", "tinyimagenet", "imagenetdogs", "imagenet10"]:
        from torchvision.datasets import ImageFolder
        dataset = ImageFolder(cfg['data']['train_path'],
                              transform=transform)
    else:
        raise ValueError('Invalid train dataset {}'.format(cfg['data']['dataset']))
    # Wrap into other dataset (__getitem__ changes)
    if augmented:  # Dataset returns an image and an augmentation of that image.
        from cdc.data.custom_dataset import AugmentedDataset
        dataset = AugmentedDataset(dataset)
    return dataset

def get_val_dataset(cfg, transform=None):
    # Base dataset
    if cfg['data']['dataset'] == 'cifar10':
        from cdc.data.cifar20_dataset import CIFAR10
        dataset = CIFAR10(cfg['data']['val_path'],
                          train='train+test', transform=transform, download=True)

    elif cfg['data']['dataset'] == 'cifar20':
        from cdc.data.cifar20_dataset import CIFAR20
        dataset = CIFAR20(cfg['data']['val_path'],
                          train='train+test', transform=transform, download=True)

    elif cfg['data']['dataset'] == 'stl10':
        from cdc.data.stl10_dataset import STL10
        dataset = STL10(cfg['data']['val_path'],
                        split='train+test', transform=transform, download=True)

    elif cfg['data']['dataset'] in ["imagenet", "imagenet100", "tinyimagenet", "imagenetdogs", "imagenet10"]:
        from torchvision.datasets import ImageFolder
        dataset = ImageFolder(cfg['data']['val_path'],
                              transform=transform)
    else:
        raise ValueError('Invalid validation dataset {}'.format(cfg['data']['dataset']))
    return dataset
def get_train_dataloader(cfg, dataset, is_drop_last = True, is_shuffle = True):
    return torch.utils.data.DataLoader(dataset, num_workers=cfg['data']['num_workers'],
            batch_size=cfg['optimizer']['batch_size'], pin_memory=True, collate_fn=collate_custom,
            drop_last=is_drop_last, shuffle=is_shuffle)
def get_val_dataloader(cfg, dataset):
    return torch.utils.data.DataLoader(dataset, num_workers=cfg['data']['num_workers'],
            batch_size=500, pin_memory=True, collate_fn=collate_custom,
            drop_last=False, shuffle=False)

def get_model(cfg, pretrain=None):
    from cdc.backbones.models import ClusteringModel
    model = ClusteringModel(cfg)
    if pretrain:
        state = torch.load(cfg['pretext_model'], map_location='cpu')
        if cfg['method'] == 'cdcv2':
            model.load_state_dict(state['state_dict'], strict=False)
        else:
            raise NotImplementedError
    elif pretrain and not os.path.exists(cfg['pretext_model']):
        raise ValueError('Path with pre-trained weights does not exist {}'.format(cfg['pretext_model']))
    else:
        pass
    model = torch.nn.DataParallel(model)
    model = model.cuda()
    return model

def get_optimizer(cfg, model1):
    params = filter(lambda p: p.requires_grad, model1.parameters())
    params = [
        {'params': model1.module.backbone.parameters(), 'lr': cfg['optimizer']['elr']},
        {'params': model1.module.cluster_head.parameters(), 'lr': cfg['optimizer']['lr']}
    ]
    if cfg['optimizer']['name'] == 'sgd':
        optimizer = torch.optim.SGD(params, **cfg['optimizer']['kwargs'])
    elif cfg['optimizer']['name'] == 'adam':
        optimizer = torch.optim.Adam(params, **cfg['optimizer']['kwargs'])
    return optimizer
