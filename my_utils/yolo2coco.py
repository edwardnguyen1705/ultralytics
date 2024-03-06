from argparse import ArgumentParser
import sys
import glob
import json
import shutil
from pathlib import Path
import os.path as osp
from tqdm import tqdm
import copy
import cv2

from file_utils import make_coco_structure
from custom_coco_info import jdict_init, START_BOUNDING_BOX_ID, PRE_DEFINE_CATEGORIES
from box_utils import to_coco_box, valid_coco_box


def convert_one_line(line, img_h, img_w):
    cls_id, x, y, w, h = line.split(" ")
    cls_id, x, y, w, h = int(cls_id), float(x), float(y), float(w), float(h)
    yolo_box = [x, y, w, h]
    box = to_coco_box(yolo_box, img_h, img_w)
    box = valid_coco_box(box, img_h, img_w)
    box.append(cls_id)
    return box

def convert_one_txt_file(txt_file, img_h, img_w):
    boxes = []
    with open(txt_file, "r") as f:
        for line in f:
            box = convert_one_line(line, img_h, img_w)
            boxes.append(box)
    return boxes

def copy_one_img(possible_img_files, coco_img_dir, new_img_id=1, use_new_img_id=False, is_cp_img=True):
    is_img_exist = False
    for img_file in possible_img_files:
        img_path = Path(img_file)
        if img_path.is_file():
            if use_new_img_id:
                _, ext = osp.splitext(img_file)
                filename = "{}{}".format(str(new_img_id).zfill(6), ext)
            else:
                filename = osp.basename(img_file)
            dst = osp.join(coco_img_dir, filename)
            if is_cp_img:
                shutil.copyfile(img_file, dst)
            is_img_exist = True
            break
    return is_img_exist, img_file, filename

def get_img_hw(img_file):
    img = cv2.imread(img_file)
    h, w, c = img.shape
    return h, w

def process_valid_img(img_file, txt_file, new_img_name, bnd_id, l_jdict):
    img_h, img_w = get_img_hw(img_file)
    img_id = int(new_img_name.split('.')[0])
    image = {
        "file_name": new_img_name,
        "height": img_h,
        "width": img_w,
        "id": img_id,
        "license": 1,
    }
    l_jdict["images"].append(image)
    boxes = convert_one_txt_file(txt_file, img_h, img_w)
    for b in boxes:
        x, y, w, h, cls_id = b
        anno = {
            "area": w * h,
            "iscrowd": 0,
            "image_id": img_id,
            "bbox": [x, y, w, h],
            "category_id": cls_id,
            "id": bnd_id,
            "segmentation": [],
        }
        l_jdict["annotations"].append(anno)
        bnd_id += 1
    return l_jdict, bnd_id


def main(args):
    splits = ['train', 'val']
    lbldir, imgdir = make_coco_structure(args.coco_dir, splits)
    bnd_id = START_BOUNDING_BOX_ID
    img_id = 1
    categories = PRE_DEFINE_CATEGORIES
    
    for split in splits:
        coco_img_dir = osp.join(imgdir, split)
        txt_dir = osp.join(args.yolo_dir, 'labels', split)
        txt_files = glob.glob(txt_dir + "/*.txt")
        jdict = copy.deepcopy(jdict_init)
        for txt_file in tqdm(txt_files):
            possible_img_files = [
                txt_file.replace('labels', 'images').replace('.txt', ext)
                for ext in [".jpg", ".png", ".jpeg"]
            ]
            is_img_exist, img_file, new_img_name = copy_one_img(possible_img_files, coco_img_dir, new_img_id=img_id, use_new_img_id=args.use_new_img_id, is_cp_img=args.is_cp_img)
            if not is_img_exist: continue
            jdict, bnd_id = process_valid_img(img_file, txt_file, new_img_name, bnd_id, jdict)
            img_id += 1
        
        for cate, cid in categories.items():
            cat = {"supercategory": "human", "id": cid, "name": cate}
            jdict["categories"].append(cat)

        json_file = osp.join(lbldir, split + '.json')
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(jdict, f, ensure_ascii=False, indent=4)

def build_argparser():
    parser = ArgumentParser(prog=__file__)
    parser.add_argument("--yolo-dir", type=str, default="YOLO", help="Path to a YOLO format dataset.")
    parser.add_argument("--coco-dir", type=str, default="COCO", help="Path to an empty dir.")
    parser.add_argument("--img-id", action="store_true", help="False: do not rename yolo images.")
    parser.add_argument("--cp-img", action="store_true", help="False: do not copy yolo images to coco folder.")
    return parser

if __name__ == "__main__":
    args = build_argparser().parse_args()
    sys.exit(main(args) or 0)
