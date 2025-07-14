import time
from itertools import product
import math
from math import sqrt
import cv2
import numpy as np
import onnxruntime
from multiprocessing import Process, Queue

from utils.box_utils import nms_numpy, after_nms_numpy
from utils.metrics_utils import APDataObject, prep_metrics

MASK_SHAPE = (138, 138, 3)

COLORS = np.array([[0, 0, 0], [244, 67, 54], [233, 30, 99], [156, 39, 176], [103, 58, 183], [100, 30, 60],
                   [63, 81, 181], [33, 150, 243], [3, 169, 244], [0, 188, 212], [20, 55, 200],
                   [0, 150, 136], [76, 175, 80], [139, 195, 74], [205, 220, 57], [70, 25, 100],
                   [255, 235, 59], [255, 193, 7], [255, 152, 0], [255, 87, 34], [90, 155, 50],
                   [121, 85, 72], [158, 158, 158], [96, 125, 139], [15, 67, 34], [98, 55, 20],
                   [21, 82, 172], [58, 128, 255], [196, 125, 39], [75, 27, 134], [90, 125, 120],
                   [121, 82, 7], [158, 58, 8], [96, 25, 9], [115, 7, 234], [8, 155, 220],
                   [221, 25, 72], [188, 58, 158], [56, 175, 19], [215, 67, 64], [198, 75, 20],
                   [62, 185, 22], [108, 70, 58], [160, 225, 39], [95, 60, 144], [78, 155, 120],
                   [101, 25, 142], [48, 198, 28], [96, 225, 200], [150, 167, 134], [18, 185, 90],
                   [21, 145, 172], [98, 68, 78], [196, 105, 19], [215, 67, 84], [130, 115, 170],
                   [255, 0, 255], [255, 255, 0], [196, 185, 10], [95, 167, 234], [18, 25, 190],
                   [0, 255, 255], [255, 0, 0], [0, 255, 0], [0, 0, 255], [155, 0, 0],
                   [0, 155, 0], [0, 0, 155], [46, 22, 130], [255, 0, 155], [155, 0, 255],
                   [255, 155, 0], [155, 255, 0], [0, 155, 255], [0, 255, 155], [18, 5, 40],
                   [120, 120, 255], [255, 58, 30], [60, 45, 60], [75, 27, 244], [128, 25, 70]], dtype='uint8')

COCO_CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                'scissors', 'teddy bear', 'hair drier', 'toothbrush')

CASTOM_CLASSES = ('first', 'backgrnd')

class Detection(Process):
    """
    Attributes
    ----------
    input_size : int
        Represents the size of the input frame.
    input : Queue
        The queue of input frames for the detection process.
    cfg : dict
        The configuration settings for the rknn-detection process. It include parameters such as 
        confidence thresholds, maximum number of output predictions, etc. (see main.py)
    q_out : Queue
        An instance of the "Queue" class with a maximum size of 3. It is used to store the 
        processed frames and prepared results for display.
    
    Methods
    -------
    permute(net_outputs)
        Permutes the elements in the net_outputs list according to a specific order.
    detect(inputs)
        Detect is the final layer of SSD. Decode location preds, apply non-maximum suppression 
        to location predictions based on conf scores and threshold to a top_k number of output 
        predictions for both confidence score and locations, as the predicted masks.
    prep_display(results)
        This method prepares the results for display. It extracts data from the inference results
        in the form: class_ids, scores, bboxes, masks
    run(None)
        Method runs in an infinite loop. It puts the frame and prepared results into the "q_out" queue.
    
    """
    
    def __init__(self, input, cfg=None):
        super().__init__(group=None, target=None, name=None, args=(), kwargs={}, daemon=True)
        self.input_size = 0
        self.input = input
        self.cfg = cfg
        self.q_out = Queue(maxsize=3)
    
    def run(self):
        while True:
            frame, inputs = self.input.get()
            results = self.detect(inputs)
            self.q_out.put((frame, results))
    
    def permute(self, net_outputs):
        '''implementation dependent'''
        pass

    def detect(self, inputs):
        '''implementation dependent'''
        pass

    def prep_display(self, results):
        '''implementation dependent'''
        pass


class RKNNDetection(Detection):
    """This class represents an implementation of the RKNNDetection algorithm, which is a subclass of the 
    Detection class. It includes methods for initializing the algorithm, permuting the network outputs, 
    performing object detection, and preparing the results for display.
    
    Attributes
    ----------
    input_size : int
        The size of the input frame.
    anchors : list
        A list of anchor boxes used for object detection.

    Methods
    -------
    __init__(input, cfg)
        Initializes the RKNNDetection algorithm by setting the input size and generating the anchor boxes.
    permute(net_outputs)
        Permutes the arrays in net_outputs to have a specific shape.
    detect(onnx_inputs)
        Performs object detection by applying non-maximum suppression.
    prep_display(results)
        Prepares the results for display.
    
    """

    def __init__(self, input, cfg):
        super().__init__(input)
        self.input_size = cfg['img_size']
        self.conf_threshold = cfg['conf_threshold']
        self.iou_threshold = cfg['iou_threshold']
    
    def filter_boxes(
        self,
        boxes: np.ndarray,
        box_confidences: np.ndarray,
        box_class_probs: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Filter boxes with object threshold."""
        box_confidences = box_confidences.flatten()
        class_max_score = np.max(box_class_probs, axis=-1)
        classes = np.argmax(box_class_probs, axis=-1)

        scores = class_max_score * box_confidences
        mask = scores >= self.conf_threshold

        return boxes[mask], classes[mask], scores[mask]

    def dfl(self, position: np.ndarray) -> np.ndarray:
        n, c, h, w = position.shape
        p_num = 4
        mc = c // p_num
        y = position.reshape(n, p_num, mc, h, w)

        exp_y = np.exp(y)
        y = exp_y / np.sum(exp_y, axis=2, keepdims=True)

        acc_metrix = np.arange(mc).reshape(1, 1, mc, 1, 1).astype(float)
        return np.sum(y * acc_metrix, axis=2)

    def box_process(self, position: np.ndarray) -> np.ndarray:
        grid_h, grid_w = position.shape[2:4]
        col, row = np.meshgrid(np.arange(grid_w), np.arange(grid_h))
        grid = np.stack((col, row), axis=0).reshape(1, 2, grid_h, grid_w)
        stride = np.array([self.input_size // grid_h, self.input_size // grid_w]).reshape(
            1, 2, 1, 1
        )

        position = self.dfl(position)
        box_xy = grid + 0.5 - position[:, 0:2, :, :]
        box_xy2 = grid + 0.5 + position[:, 2:4, :, :]
        xyxy = np.concatenate((box_xy * stride, box_xy2 * stride), axis=1)

        return xyxy

    def detect(self, inputs):
        '''
        Inputs (boxes_classes_ scores Tensor)
        Returns (boxes, classes, scores) | (None, None, None)
        -------
        
        '''
        def sp_flatten(_in):
            ch = _in.shape[1]
            return _in.transpose(0, 2, 3, 1).reshape(-1, ch)

        def add_offset(boxes):
            n_boxes = boxes.shape[0]
            step_ind = n_boxes//6
            x_offset = 640
            y_offset = 540
            #boxes[0][:,0] # first crop have not offset
            boxes[1*step_ind:2*step_ind,0] += x_offset # second crop have x_offset only
            boxes[1*step_ind:2*step_ind,2] += x_offset # second crop have x_offset only
            boxes[2*step_ind:3*step_ind,0] += 2*x_offset # third crop have x_offset only
            boxes[2*step_ind:3*step_ind,2] += 2*x_offset # third crop have x_offset only
            boxes[3*step_ind:4*step_ind,1] += y_offset #  crop have y_offset only
            boxes[3*step_ind:4*step_ind,3] += y_offset #  crop have y_offset only
            boxes[4*step_ind:5*step_ind,0] += x_offset #  crop have x_ and y_offset 
            boxes[4*step_ind:5*step_ind,2] += x_offset # crop have x_ and y_offset
            boxes[4*step_ind:5*step_ind,1] += y_offset #  crop have x_ and y_offset 
            boxes[4*step_ind:5*step_ind,3] += y_offset # crop have x_ and y_offset 
            boxes[5*step_ind:,0] += 2*x_offset #  crop have x_ and y_offset 
            boxes[5*step_ind:,2] += 2*x_offset # crop have x_ and y_offset
            boxes[5*step_ind:,1] += y_offset #  crop have x_ and y_offset 
            boxes[5*step_ind:,3] += y_offset # crop have x_ and y_offset
            return boxes

        defualt_branch = 3
        pair_per_branch = len(inputs) // defualt_branch

        boxes, classes_conf, scores = [], [], []
        for i in range(defualt_branch):
            boxes.append(self.box_process(inputs[pair_per_branch * i]))
            classes_conf.append(sp_flatten(inputs[pair_per_branch * i + 1]))
            scores.append(np.ones_like(classes_conf[-1][:, :1], dtype=np.float32))

        boxes = [sp_flatten(b) for b in boxes]
        boxes = [add_offset(b) for b in boxes]
        boxes = np.concatenate(boxes)
        classes_conf = np.concatenate(classes_conf)
        scores = np.concatenate(scores).flatten()

        boxes, classes, scores = self.filter_boxes(boxes, scores, classes_conf)
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(), scores.tolist(), self.conf_threshold, self.iou_threshold
        )
        if isinstance(indices, tuple):
            return None, None, None

        boxes = boxes[indices]
        classes = classes[indices]
        scores = scores[indices]

        return boxes, classes, scores
    
    def prep_display(self, results):
        boxes, indices, confidences = results
        ids_p = []
        class_p = []
        box_p = []
        
        for i in indices:
            x_center, y_center, w, h = boxes[i]
            x1 = int((x_center - w / 2))
            y1 = int((y_center - h / 2))
            x2 = int((x_center + w / 2))
            y2 = int((y_center + h / 2))
            conf = confidences[i]
            
            # Create dummy data for missing parameters
            ids_p.append(0)  # Class ID (0 for 'first')
            class_p.append(conf)  # Confidence score
            box_p.append([x1, y1, x2, y2])  # Bounding box
        
        # Create dummy masks (since we're doing detection, not segmentation)        
        return (
            np.array(ids_p), 
            np.array(class_p), 
            np.array(box_p), 
        )


class PostProcess():
    """Class to handle post-processing of yolact inference results.

    Attributes
    ----------
    detection : Detection
        Detection class object.

    Methods
    -------
    run()
        Starts the detection process.
    get_outputs()
        Retrieves the prepared results from the detection process.
        
    """
    
    def __init__(self, queue, cfg:None):
        """
        Parameters
        ----------
        queue : Queue
            An instance of the "Queue" class with a maximum size of 3, used to store processed frames 
            and prepared results for display.
        cfg : dict
            Configuration settings for the detection process. May include parameters such as 
            confidence thresholds, maximum number of output predictions, etc. Default is None.
        onnx : bool
            Flag indicating whether to use ONNXDetection or RKNNDetection. Default is True.
        """
        self.detection = RKNNDetection(queue, cfg)
    
    def run(self):
        self.detection.start()
    
    def get_outputs(self):
        return self.detection.q_out.get()


class Visualizer():
    
    def __init__(self):
        pass

    def show_results(self, frame, out):
        """
        Save the given frame on the screen with bounding boxes
        """
        # Remove batch dimension if present
        if frame.ndim == 4 and frame.shape[0] == 1:
            frame = frame[0]
            
        if frame is None:
            return
        
        dwdh = (2, 2)
        ratio = 1
        # Convert to BGR for OpenCV
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # Unpack outputs
        boxes, classes, scores = out
        if all(x is not None for x in (boxes, classes, scores)):
             boxes -= np.array(dwdh * 2)
             boxes /= ratio
             boxes = boxes.round().astype(np.int32)
        
        # Draw bounding boxes
        for box, score, cl in zip(boxes, scores, classes):
            top, left, right, bottom = map(int, box)
            cv2.rectangle(img=frame, pt1=(top, left), pt2=(right, bottom), color=(255, 0, 0),
                          thickness=2,)
            text_str = f"{CASTOM_CLASSES[cl]} {score:.2f}"
            cv2.putText(img=frame, text=text_str, org=(top, left - 6),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(0, 0, 255), thickness=2,)
        
        cv2.imshow('Yolo_v8 Inference', frame)
        cv2.waitKey(1)
