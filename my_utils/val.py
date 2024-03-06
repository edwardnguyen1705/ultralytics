from argparse import ArgumentParser
import sys

from ultralytics import YOLO


def main(args):
    model = YOLO(args.weight)
    validation_results = model.val(data=args.data_cfg,
                                imgsz=args.imgsz,
                                batch=args.batch,
                                save_json=args.save_json,
                                # conf=0.25,
                                # iou=0.6,
                                device=args.device)

def build_argparser():
    parser = ArgumentParser(prog=__file__)
    parser.add_argument(
        "--weight",
        type=str,
        default="yolov8n.pt",
        help="Path to model.pt.",
    )
    parser.add_argument(
        "--data-cfg",
        type=str,
        default="ultralytics/cfg/datasets/coco.yaml",
        help="Path to data cfg.",
    )
    parser.add_argument(
        "--imgsz", type=int, default=640, help="inference size (pixels)"
    )
    parser.add_argument("--batch", type=int, default=1, help="batch size")
    parser.add_argument("--device", type=str, default='0', help="GPU device")
    parser.add_argument("--save-json", action="store_true", help="Save predictions to a json file.")

    return parser

if __name__ == "__main__":
    args = build_argparser().parse_args()
    sys.exit(main(args) or 0)
