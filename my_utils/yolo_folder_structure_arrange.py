"""
Convert
from nearly yolo folder structure
|-- images
|-- labels
|-- train.txt
`-- val.txt
to yolo folder structure
|-- images
|   |-- train
|   `-- val
|-- labels
|   |-- train
|   `-- val
|-- train.txt
`-- val.txt
"""

import os.path as osp
from pathlib import Path
import shutil

from file_utils import make_yolo_structure

ds_bdir_in = '/usr/datasets/ds_name'
ds_bdir_out = '/usr/datasets/ds_name_yolo'
splits = ['val', 'train']

lbldir_out, imgdir_out = make_yolo_structure(ds_bdir_out, splits)
print(f'lbldir_out: {lbldir_out}, imgdir_out: {imgdir_out}')

for split in splits:
    split_txt = osp.join(ds_bdir_in, split + '.txt')
    print(f'Processing {split_txt}')
    img_names = []
    with open(split_txt, "r") as f:
        for line in f:
            line = line.rstrip()
            img_name = osp.basename(line)
            name = Path(img_name).stem
            txt_name = name + '.txt'
            txt_path = osp.join(ds_bdir_in, 'labels', txt_name)
            img_path = osp.join(ds_bdir_in, 'images', img_name)
            txt_dir_dst = osp.join(lbldir_out, split)
            img_dir_dst = osp.join(imgdir_out, split)
            shutil.move(txt_path, txt_dir_dst)
            shutil.move(img_path, img_dir_dst)
            img_names.append(osp.join('./images', split, img_name))
    
    split_txt_out = osp.join(ds_bdir_out, f"{split}.txt")
    with open(split_txt_out, "w") as f:
        for item in img_names:
            f.write("%s\n" % item)
            