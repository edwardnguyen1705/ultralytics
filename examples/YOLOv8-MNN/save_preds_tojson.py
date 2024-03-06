# Ultralytics YOLO 🚀, AGPL-3.0 license

import argparse

import os.path as osp
from pathlib import Path
import glob
import json
import cv2
from tqdm import tqdm

import numpy as np
import MNN
from ultralytics.utils import ASSETS, yaml_load
from ultralytics.utils.checks import check_requirements, check_yaml

class Network:
    def __init__(self):
        self.interpreter = None

    def load_model(self, model_path):
        self.interpreter = MNN.Interpreter(model_path)
        self.session = self.interpreter.createSession()

    def inference(self, input_size, output_name, mnn_img):
        input_tensor = self.interpreter.getSessionInput(self.session)
        tmp_input = MNN.Tensor(((1,3,input_size[1], input_size[0])), MNN.Halide_Type_Float,\
                        mnn_img, MNN.Tensor_DimensionType_Caffe)
        input_tensor.copyFrom(tmp_input)
        self.interpreter.runSession(self.session)
        output_tensor = self.interpreter.getSessionOutput(self.session, output_name)
        output_shape = output_tensor.getShape()
        output = np.array(output_tensor.getData(), dtype=float).reshape(output_shape)
        return output

class YOLOv8:
    """YOLOv8 object detection model class for handling inference."""
    def __init__(self, mnn_model, confidence_thres, iou_thres, input_width, input_height, save_pred_dir, output_name='output0', data_cfg='ultralytics/cfg/datasets/coco.yaml'):
        """
        Initializes an instance of the YOLOv8 class.

        Args:
            mnn_model: Path to the MNN model.
            input_image: Path to the input image.
            confidence_thres: Confidence threshold for filtering detections.
            iou_thres: IoU (Intersection over Union) threshold for non-maximum suppression.
        """
        self.mnn_model = mnn_model
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres

        # Load the class names from the COCO dataset
        self.classes = yaml_load(check_yaml(data_cfg))["names"]

        # Generate a color palette for the classes
        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))
        
        self.input_width = input_width
        self.input_height = input_height
        
        self.jdict = []
        self.save_dir = save_pred_dir
        self.session = Network()
        self.session.load_model(self.mnn_model)
        self._output_names = output_name

    def preprocess(self, img_file):
        """
        Preprocesses the input image before performing inference.

        Returns:
            image_data: Preprocessed image data ready for inference.
        """
        # Read the input image using OpenCV
        img = cv2.imread(img_file)
        # Get the height and width of the input image
        self.img_height, self.img_width = img.shape[:2]

        # Convert the image color space from BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize the image to match the input shape
        img = cv2.resize(img, (self.input_width, self.input_height))

        # Normalize the image data by dividing it by 255.0
        image_data = np.array(img) / 255.0

        # Transpose the image to have the channel dimension as the first dimension
        image_data = np.transpose(image_data, (2, 0, 1))  # Channel first

        # Expand the dimensions of the image data to match the expected input shape
        image_data = np.expand_dims(image_data, axis=0).astype(np.float32)

        # Return the preprocessed image data
        return image_data

    def postprocess(self, output):
        """
        Performs post-processing on the model's output to extract bounding boxes, scores, and class IDs.

        Args:
            output (numpy.ndarray): The output of the model.

        Returns:
            numpy.ndarray: The input image with detections drawn on it.
        """

        # Transpose and squeeze the output to match the expected shape
        outputs = np.transpose(np.squeeze(output[0]))

        # Get the number of rows in the outputs array
        rows = outputs.shape[0]

        # Lists to store the bounding boxes, scores, and class IDs of the detections
        boxes = []
        scores = []
        class_ids = []

        # Calculate the scaling factors for the bounding box coordinates
        x_factor = self.img_width / self.input_width
        y_factor = self.img_height / self.input_height

        # Iterate over each row in the outputs array
        for i in range(rows):
            # Extract the class scores from the current row
            classes_scores = outputs[i][4:]

            # Find the maximum score among the class scores
            max_score = np.amax(classes_scores)

            # If the maximum score is above the confidence threshold
            if max_score >= self.confidence_thres:
                # Get the class ID with the highest score
                class_id = np.argmax(classes_scores)

                # Extract the bounding box coordinates from the current row
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

                # Calculate the scaled coordinates of the bounding box
                left = int((x - w / 2) * x_factor)
                top = int((y - h / 2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)

                # Add the class ID, score, and box coordinates to the respective lists
                class_ids.append(class_id)
                scores.append(max_score)
                boxes.append([left, top, width, height])

        # Apply non-maximum suppression to filter out overlapping bounding boxes
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_thres, self.iou_thres)

        # Iterate over the selected indices after non-maximum suppression
        predictions = []
        for i in indices:
            # Get the box, score, and class ID corresponding to the index
            box = boxes[i]
            score = scores[i]
            class_id = class_ids[i]
            left, top, width, height = box
            predictions.append([left, top, width, height, score, class_id])
        return predictions

    def pred_to_json(self, predictions, filename):
        """Serialize YOLO predictions to COCO json format."""
        stem = Path(filename).stem
        image_id = int(stem) if stem.isnumeric() else stem
        for p in predictions:
            b = p[:4]
            self.jdict.append(
                {
                    "image_id": image_id,
                    "category_id": int(p[5]),
                    "bbox": [round(x, 3) for x in b],
                    "score": round(p[4], 5),
                }
            )
            
    def detect_img(self, img_file):
        img_data = self.preprocess(img_file)
        outputs = self.session.inference((self.input_width, self.input_height), self._output_names, img_data)
        predictions = self.postprocess(outputs)
        self.pred_to_json(predictions, img_file)

    def detect_imgs(self, img_files):
        for img_file in tqdm(img_files):
            self.detect_img(img_file)
        
        with open(osp.join(self.save_dir,"predictions.json"), "w") as f:
            json.dump(self.jdict, f)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="yolov8n.mnn", help="Input your MNN model.")
    parser.add_argument("--data-cfg", type=str, default="ultralytics/cfg/datasets/coco.yam", help="Path to a data cfg.")
    parser.add_argument("--conf-thres", type=float, default=0.001, help="Confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.6, help="NMS IoU threshold")
    parser.add_argument("--input-size", type=int, default=640, help="Input size")
    parser.add_argument("--img-dir", type=str, default="./", help="Images dir.")
    parser.add_argument("--save-pred-dir", type=str, default="./", help="Save predictions dir.")
    args = parser.parse_args()

    detector = YOLOv8(args.model, args.conf_thres, args.iou_thres, args.input_size, args.input_size, args.save_pred_dir, output_name='output0', data_cfg=args.data_cfg)

    types = ('*.png', '*.jpg', '*.jpeg')
    img_files = []
    for t in types:
        img_files.extend(glob.glob(osp.join(args.img_dir, t)))
    
    detector.detect_imgs(img_files)
