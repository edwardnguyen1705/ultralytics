# YOLOv8 - MNN

## ONN export

```bash
$ python examples/YOLOv8-MNN/export.py --weight yolov8n.pt --format onnx --imgsz 256 --half --simplify
```

## MNN export

```bash
$ python -m MNN.tools.mnnconvert -f ONNX --modelFile model.onnx --MNNModel model.mnn --bizCode yolov8n256 --optimizePrefer 2
```

```bash
$ 
```

This project implements YOLOv8 using MNN.

## Installation

To run this project, you need to install the required dependencies. The following instructions will guide you through the installation process.

### Installing Required Dependencies

You can install the required dependencies by running the following command:

```bash
$ pip install MNN
```

### Usage

After successfully installing the required packages, you can run the YOLOv8 implementation using the following command:

```bash
$ python main.py --model yolov8n.mnn --data-cfg ultralytics/cfg/datasets/coco.yam --img image.jpg --conf-thres 0.5 --iou-thres 0.5
```

Make sure to replace yolov8n.mnn with the path to your YOLOv8 MNN model file, image.jpg with the path to your input image, and adjust the confidence threshold (conf-thres) and IoU threshold (iou-thres) values as needed.
