This is official code for ICLR25 paper "Towards Calibrated Deep Clustering Network".

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
[Pretrained Model and Trained Model](https://seunic-my.sharepoint.cn/:f:/g/personal/220222092_seu_edu_cn/EvPQ5Lq6q5pDgoBQOW1Sr-cBNYwh7Ez89QB8tf_XdrcJnw?e=BQ8g06)

## Citation
If you find this project useful for your research, please consider citing our paper:

```latex
@inproceedings{cdc2025,
  title={Towards Calibrated Deep Clustering Network},
  author={Jia, Yuheng and Cheng, Jianhong and Liu, Hui and Hou, Junhui},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025}
}
```
For any questions, please add issues or contact the main authors.

## Acknowledgment

[solo-learn](https://github.com/vturrisi/solo-learn)

[SCAN](https://github.com/wvangansbeke/Unsupervised-Classification)

[SPICE](https://github.com/niuchuangnn/SPICE)


