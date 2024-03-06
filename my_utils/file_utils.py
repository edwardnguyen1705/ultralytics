import os.path as osp
import shutil
from pathlib import Path


def make_dir(new_dir, parents=True, exist_ok=True, rm=False):
    if rm and osp.exists(new_dir) and osp.isdir(new_dir):
        shutil.rmtree(new_dir)
    Path(new_dir).mkdir(parents=True, exist_ok=True)

def make_voc_structure(datadir, parents=True, exist_ok=True, rm=False):
    lbl_dir = osp.join(datadir, 'Annotations')
    img_dir = osp.join(datadir, 'JPEGImages')
    tmp_dir = osp.join(datadir, 'ImageSets')
    split_dir = osp.join(tmp_dir, 'Main')

    [make_dir(new_dir, parents=parents, exist_ok=exist_ok, rm=rm)
     for new_dir in [datadir, lbl_dir, img_dir, tmp_dir, split_dir]]

    return lbl_dir, img_dir, split_dir


def make_yolo_structure(datadir, splits, parents=True, exist_ok=True, rm=False):
    lbldir = osp.join(datadir, 'labels')
    imgdir = osp.join(datadir, 'images')

    [make_dir(new_dir, parents=parents, exist_ok=exist_ok, rm=rm)
     for new_dir in [datadir, lbldir, imgdir]]
    
    dirs = []
    for split in splits:
        split_lbl_dir = osp.join(lbldir, split)
        split_img_dir = osp.join(imgdir, split)
        dirs.append(split_lbl_dir)
        dirs.append(split_img_dir)

    [make_dir(d, parents=parents, exist_ok=exist_ok, rm=rm) for d in dirs]

    return lbldir, imgdir

def make_coco_structure(datadir, splits, parents=True, exist_ok=True, rm=False):
    lbldir = osp.join(datadir, 'annotations')
    imgdir = osp.join(datadir, 'images')

    [make_dir(new_dir, parents=parents, exist_ok=exist_ok, rm=rm)
     for new_dir in [datadir, lbldir, imgdir]]
    
    dirs = []
    for split in splits:
        split_img_dir = osp.join(imgdir, split)
        dirs.append(split_img_dir)

    [make_dir(d, parents=parents, exist_ok=exist_ok, rm=rm) for d in dirs]

    return lbldir, imgdir


def read_txt(txt_path):
    with open(str(txt_path), 'r', encoding='utf-8') as f:
        data = list(map(lambda x: x.rstrip('\n'), f))
    return data

def is_path_exists(file_path):
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f'The {file_path} is not exists!!!')
