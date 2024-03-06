from argparse import ArgumentParser
import sys
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def eval_json(pred_json, anno_json, eval_each_cls=False):
    """Evaluates YOLO output in JSON format and returns performance statistics."""
    anno = COCO(anno_json)
    cats = anno.getCatIds()
    pred = anno.loadRes(pred_json)
    eval = COCOeval(anno, pred, "bbox")
    
    print(80 * "*")
    print(f'Eval all classes')
    eval.evaluate()
    eval.accumulate()
    eval.summarize()
    
    if eval_each_cls:
        for cat in cats:
            print(80 * "*")
            print(f'Eval class id: {cat}')
            eval.params.catIds = [cat]
            eval.evaluate()
            eval.accumulate()
            eval.summarize()

def main(args):
    eval_json(args.pred_json, args.anno_json, eval_each_cls=args.eval_each_cls)

def build_argparser():
    parser = ArgumentParser(prog=__file__)
    parser.add_argument(
        "--pred-json",
        type=str,
        default="./predictions.json",
        help="Path to predictions json.",
    )
    parser.add_argument(
        "--anno-json",
        type=str,
        default="./annotations.json",
        help="Path to annotations json.",
    )

    parser.add_argument("--eval-each-cls", action="store_true", help="Evaluate each class.")

    return parser


if __name__ == "__main__":
    args = build_argparser().parse_args()
    sys.exit(main(args) or 0)
