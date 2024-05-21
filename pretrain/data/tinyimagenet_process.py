'''
@File  :tinyimagenet_process.py
@Date  :2023/1/25 16:13
@Desc  :process tinyimagenet valdatasets
'''
import os
import shutil

print(os.getcwd())

read_path = "../datasets/tiny-imagenet-200/val/images"

save_path = "../datasets/tiny-imagenet-200/val/images_new"
if not os.path.exists(save_path):
    os.mkdir(save_path)
fileType = '.JPEG'

with open('../datasets/tiny-imagenet-200/val/val_annotations.txt', 'r') as f:
    for name in f:
        fileName = name.strip().split('\t')
        for file in os.listdir(read_path):
            if fileName[0] == file:
                if not os.path.exists(save_path+'/' + fileName[1]):
                    os.mkdir(save_path+'/' + fileName[1])
                shutil.copy(os.path.join(read_path, fileName[0]), save_path+'/' + fileName[1])