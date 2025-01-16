"""
https://github.com/ibaiGorordo/ONNX-YOLOv8-Object-Detection/tree/main
"""


import cv2
import numpy as np
import onnxruntime as ort

from object_detection.detectors.detector import Detector


class ONNXDetector(Detector):

    def __init__(self, path: str, conf: float) -> None:
        super().__init__(path, conf)
        self.iou_threshold = 0.5
        self._initialize_model(path)


    def detect(self, image: np.ndarray) -> tuple:
        input_tensor = self._prepare_input(image)
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})
        return self._process_output(outputs)


    def _initialize_model(self, path):
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = 2
        session_options.inter_op_num_threads = 1
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(
            path,
            providers=["CPUExecutionProvider"],
            sess_options=session_options
        )
        self._get_input_details()
        self._get_output_details()


    def _prepare_input(self, image: np.ndarray):
        self.img_height, self.img_width = image.shape[:2]
        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_img = cv2.resize(input_img, (self.input_width, self.input_height))
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)
        return input_tensor


    def _process_output(self, output):
        predictions = np.squeeze(output[0]).T

        # Filter out object confidence scores below threshold
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > self.conf, :]
        scores = scores[scores > self.conf]

        if len(scores) == 0:
            return [], [], []

        # Get the class with the highest confidence
        class_ids = np.argmax(predictions[:, 4:], axis=1)
        boxes = self._extract_boxes(predictions)
        indices = self._multiclass_nms(boxes, scores, class_ids, self.iou_threshold)

        boxes = boxes[indices].tolist()
        scores = scores[indices].astype(float).tolist()
        class_ids = class_ids[indices].astype(int).tolist()

        return boxes, scores, class_ids


    def _extract_boxes(self, predictions):
        boxes = predictions[:, :4]
        boxes = self._rescale_boxes(boxes)
        boxes = self._xywh2xyxy(boxes)
        return boxes


    def _rescale_boxes(self, boxes):
        input_shape = np.array([self.input_width, self.input_height, self.input_width, self.input_height])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([self.img_width, self.img_height, self.img_width, self.img_height])
        return boxes


    def _xywh2xyxy(self, x):
        y = np.copy(x)
        y[..., 0] = x[..., 0] - x[..., 2] / 2
        y[..., 1] = x[..., 1] - x[..., 3] / 2
        y[..., 2] = x[..., 0] + x[..., 2] / 2
        y[..., 3] = x[..., 1] + x[..., 3] / 2
        return y


    def _compute_iou(self, box, boxes):
        # Compute xmin, ymin, xmax, ymax for both boxes
        xmin = np.maximum(box[0], boxes[:, 0])
        ymin = np.maximum(box[1], boxes[:, 1])
        xmax = np.minimum(box[2], boxes[:, 2])
        ymax = np.minimum(box[3], boxes[:, 3])

        # Compute intersection area
        intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

        # Compute union area
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union_area = box_area + boxes_area - intersection_area

        # Compute IoU
        iou = intersection_area / union_area
        return iou


    def _nms(self, boxes, scores, iou_threshold):
        # Sort by score
        sorted_indices = np.argsort(scores)[::-1]

        keep_boxes = []
        while sorted_indices.size > 0:
            # Pick the last box
            box_id = sorted_indices[0]
            keep_boxes.append(box_id)

            # Compute IoU of the picked box with the rest
            ious = self._compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])

            # Remove boxes with IoU over the threshold
            keep_indices = np.where(ious < iou_threshold)[0]

            # print(keep_indices.shape, sorted_indices.shape)
            sorted_indices = sorted_indices[keep_indices + 1]

        return keep_boxes


    def _multiclass_nms(self, boxes, scores, class_ids, iou_threshold):
        unique_class_ids = np.unique(class_ids)

        keep_boxes = []
        for class_id in unique_class_ids:
            class_indices = np.where(class_ids == class_id)[0]
            class_boxes = boxes[class_indices,:]
            class_scores = scores[class_indices]

            class_keep_boxes = self._nms(class_boxes, class_scores, iou_threshold)
            keep_boxes.extend(class_indices[class_keep_boxes])

        return keep_boxes


    def _get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]
        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]


    def _get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]
