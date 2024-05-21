'''
@File  :model.py
@Date  :2023/1/29 20:01
@Desc  :
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from cdc.backbones.resnet import resnet18, resnet34, resnet50

class ClusteringModel(nn.Module):
    def __init__(self, cfg):
        super(ClusteringModel, self).__init__()
        if cfg['backbone']['name'].startswith("resnet"):
            if cfg['backbone']['name'] == "resnet18":
                self.backbone = resnet18(cfg['method'])
            elif cfg['backbone']['name'] == "resnet34":
                self.backbone = resnet34(cfg['method'])
            elif cfg['backbone']['name'] == "resnet50":
                self.backbone = resnet50(cfg['method'])
            self.backbone.fc = nn.Identity()
            self.backbone_dim = self.backbone.inplanes
            cifar = cfg['data']['dataset'] in ["cifar10", "cifar20", "cifar100"]
            if cifar:
                self.backbone.conv1 = nn.Conv2d(
                    3, 64, kernel_size=3, stride=1, padding=1, bias=False
                )
                self.backbone.maxpool = nn.Identity()
        self.nheads = cfg['backbone']['nheads']
        assert (isinstance(self.nheads, int))
        assert (self.nheads > 0)
        self.cluster_head = nn.ModuleList([
            nn.Sequential(
                          nn.Linear(self.backbone_dim, 512),
                          nn.BatchNorm1d(512),
                          nn.ReLU(inplace=True),
                        nn.Linear(512, cfg['backbone']['nclusters'])
        ) for _ in range(self.nheads)])

    def forward(self, x, forward_pass='default', dropout = None):
        if forward_pass == 'default':
            out = [cluster_head(self.backbone(x)) for cluster_head in self.cluster_head]

        elif forward_pass == 'backbone':
            out = self.backbone(x)

        elif forward_pass == 'head':
            out = [cluster_head(x) for cluster_head in self.cluster_head]

        elif forward_pass == 'return_all':
            x = self.backbone(x)
            if dropout is not None:
                x = nn.Dropout(dropout)(x)
            out = {'features': x,
                       'output': [cluster_head(x) for cluster_head in self.cluster_head]}

        else:
            raise ValueError('Invalid forward pass {}'.format(forward_pass))

        return out
class CaliMLP(nn.Module):
    def __init__(self, cfg):
        super(CaliMLP, self).__init__()
        self.calibration_head = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, cfg['backbone']['nclusters'])
        )
    def forward(self, x, forward_pass=None):
        if forward_pass == 'calibration':
            out = self.calibration_head(x)
        else:
            raise ValueError('Invalid forward pass {}'.format(forward_pass))
        return out