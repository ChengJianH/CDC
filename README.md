
## Requirements
* torch
* torchvision
* tqdm
* einops
* wandb
* pytorch-lightning
* lightning-bolts
* torchmetrics
* scipy
* timm
* omegaconf
* munkres
* easydict

**Optional**:
* nvidia-dali
* matplotlib
* seaborn
* pandas
* umap-learn

## Training
### A. Pretrain
This part is cloned from solo-learn (a library of self-supervised methods). For pretraining the backbone, follow one of the many bash files in `scripts/pretrain/`.

```bash
python main_pretrain.py \
    # path to training script folder
    --config-path scripts/pretrain/cifar/ \
    # training config name
    --config-name mocov2plus.yaml
```
### B. CDC Training
Follow one of the many bash files in `scripts/cdc/` to run CDC method.

```bash
python main_cdc.py \
    # path to config training environment
    --config_env scripts/cdc/env.yaml \    
    # training config name
    --config_exp scripts/cdc/cifar10/cdc_s100_c500_lr0001_bs1000_su500.yaml
```

**Note**: GPU memory of all the experiments are limited in 10GB.

## Model Zoo
Comming soon.

## Acknowledgment

[solo-learn](https://github.com/vturrisi/solo-learn)

[SCAN](https://github.com/wvangansbeke/Unsupervised-Classification)

[SPICE](https://github.com/niuchuangnn/SPICE)
