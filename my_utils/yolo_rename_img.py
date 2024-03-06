"""
Rename images and txts to be int so it is easier to be converted to COCO format.
"""

from argparse import ArgumentParser
import sys
import os.path as osp
from tqdm import tqdm
import shutil
from pathlib import Path

from file_utils import make_yolo_structure

def main(args):
    splits = ['train', 'val']
    lbldir_out, imgdir_out = make_yolo_structure(args.yolo_rename_dir, splits)
    print(f'lbldir_out: {lbldir_out}, imgdir_out: {imgdir_out}')

    img_id = 1
    for split in splits:
        split_txt = osp.join(args.yolo_dir, split + '.txt')
        print(f'Processing {split_txt}')
        img_names = []
        with open(split_txt, "r") as f:
            for line in tqdm(f):
                line = line.rstrip()
                img_name = osp.basename(line)
                name = Path(img_name).stem
                txt_name = name + '.txt'
                
                txt_path = osp.join(args.yolo_dir, 'labels', split, txt_name)
                img_path = osp.join(args.yolo_dir, 'images', split, img_name)
                
                txt_dir_dst = osp.join(lbldir_out, split)
                img_dir_dst = osp.join(imgdir_out, split)
                
                _, ext = osp.splitext(img_name)
                newname_wo_ext = str(img_id).zfill(6)
                newname_img = newname_wo_ext + ext
                newname_txt = newname_wo_ext + '.txt'
                
                shutil.copyfile(img_path, osp.join(img_dir_dst, newname_img))
                shutil.copyfile(txt_path, osp.join(txt_dir_dst, newname_txt))
                
                img_names.append(osp.join('./images', split, newname_img))
            
                img_id += 1
        split_txt_out = osp.join(args.yolo_rename_dir, f"{split}.txt")
        with open(split_txt_out, "w") as f:
            for item in img_names:
                f.write("%s\n" % item)

def build_argparser():
    parser = ArgumentParser(prog=__file__)
    parser.add_argument("--yolo-dir", type=str, default="YOLO", help="Path to a YOLO format dataset.")
    parser.add_argument("--yolo-rename-dir", type=str, default="YOLO", help="Path to an empty dir.")
    return parser

if __name__ == "__main__":
    args = build_argparser().parse_args()
    sys.exit(main(args) or 0)