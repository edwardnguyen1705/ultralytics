from argparse import ArgumentParser
import os
import sys
from pathlib import Path
from pycocotools.coco import COCO
import cv2


cls_names = ["head", "face", "cellphone"]

def draw_bbox(img, bbox, cls_name):
    bbox = list(map(int, bbox))
    xmin, ymin, width, height = bbox
    xmax = xmin + width
    ymax = ymin + height
    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), thickness=1)
    cv2.putText(
        img, str(cls_name), (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0)
    )

def viz_annos(json_path, img_dir, save_dir, num_imgs, isShow=False, delay=1000):
    coco = COCO(json_path)
    cats = coco.loadCats(coco.getCatIds())
    if isShow:
        windown_name = "COCO Viz"
        cv2.namedWindow(windown_name, cv2.WINDOW_NORMAL)
    
    i = 0
    for k, v in coco.imgs.items():
        img_file = os.path.join(img_dir, v["file_name"])
        img = cv2.imread(img_file)
        for anno in coco.imgToAnns[k]:
            bbox = anno["bbox"]
            # cls_idx = cls_indices.index(anno["category_id"]) # COCO2017
            cls_idx = anno["category_id"]
            cls_name = cls_names[cls_idx]
            draw_bbox(img, bbox, cls_name)
        if isShow:
            print("Please press 'q' to quit")
            cv2.imshow(windown_name, img)
            if cv2.waitKey(delay) & 0xFF == ord("q"):
                break
        else:
            img_file = os.path.join(save_dir, v["file_name"])
            cv2.imwrite(img_file, img)
        i += 1
        if i > num_imgs: break

def main(args):
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    splits = ["train", "val"]
    for split in splits:
        json_path = os.path.join(args.data_root, "annotations", split + ".json")
        img_root = os.path.join(args.data_root, "images", split)
        viz_annos(json_path, img_root, args.save_dir, args.num_imgs, isShow=args.show, delay=args.delay_time)
    
def build_argparser():
    parser = ArgumentParser(prog=__file__)
    parser.add_argument("--data-root", type=str, default="COCO", help="Path to a COCO format dataset.")
    parser.add_argument("--save-dir", type=str, default="./", help="Folder to save viz images.")
    parser.add_argument("--num-imgs", type=int, default=10, help="Num. of viz imgs.")
    parser.add_argument("--delay-time", type=int, default=1000, help="Delay time for cv2.waitKey.")
    parser.add_argument("--show", action="store_true", help="imgshow.")

    return parser

if __name__ == "__main__":
    args = build_argparser().parse_args()
    sys.exit(main(args) or 0)
