from argparse import ArgumentParser
import sys

from ultralytics import YOLO


def main(args):
    model = YOLO(args.weight)
    model.export(imgsz=args.imgsz, format=args.format, half=args.half, dynamic=args.dynamic, simplify=args.simplify, verbose=False)

def build_argparser():
    parser = ArgumentParser(prog=__file__)
    parser.add_argument("--weight", type=str, default="yolov8n.pt", help="Path to model.pt.")
    parser.add_argument("--format", type=str, default="onnx", help="Target format.")
    parser.add_argument("--imgsz", type=int, default=640, help="inference size (pixels).")
    parser.add_argument("--half", action="store_true", help="Enables FP16 (half-precision) quantization, reducing model size and potentially speeding up inference on supported hardware..")
    parser.add_argument("--dynamic", action="store_true", help="Allows dynamic input sizes for ONNX and TensorRT exports, enhancing flexibility in handling varying image dimensions.")
    parser.add_argument("--simplify", action="store_true", help="	Simplifies the model graph for ONNX exports, potentially improving performance and compatibility.")
    return parser

if __name__ == "__main__":
    args = build_argparser().parse_args()
    sys.exit(main(args) or 0)
