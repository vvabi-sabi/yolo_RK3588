import torch
import cv2
import numpy as np
from py_utils.coco_utils import COCO_test_helper
from multiprocessing import Process, Queue


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

CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
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
        self.input_size = cfg['input_size']
        self.obj_threshold = cfg['obj_threshold']
        self.nms_threshold = cfg['nms_threshold']

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
        max_det, nc = 300, len(CLASSES)

        boxes, scores = [], []
        defualt_branch=3
        pair_per_branch = len(inputs)//defualt_branch
        # Python 忽略 score_sum 输出
        for i in range(defualt_branch):
            boxes.append(self.box_process(inputs[pair_per_branch*i]))
            scores.append(inputs[pair_per_branch*i+1])

        def sp_flatten(_in):
            ch = _in.shape[1]
            _in = _in.transpose(0,2,3,1)
            return _in.reshape(-1, ch)

        boxes = [sp_flatten(_v) for _v in boxes]
        scores = [sp_flatten(_v) for _v in scores]

        boxes = torch.from_numpy(np.expand_dims(np.concatenate(boxes), axis=0))
        scores = torch.from_numpy(np.expand_dims(np.concatenate(scores), axis=0))

        max_scores = scores.amax(dim=-1)
        max_scores, index = torch.topk(max_scores, max_det, axis=-1)
        index = index.unsqueeze(-1)
        boxes = torch.gather(boxes, dim=1, index=index.repeat(1, 1, boxes.shape[-1]))
        scores = torch.gather(scores, dim=1, index=index.repeat(1, 1, scores.shape[-1]))

        scores, index = torch.topk(scores.flatten(1), max_det, axis=-1)
        labels = index % nc
        index = index // nc
        boxes = boxes.gather(dim=1, index=index.unsqueeze(-1).repeat(1, 1, boxes.shape[-1]))

        preds = torch.cat([boxes, scores.unsqueeze(-1), labels.unsqueeze(-1)], dim=-1)

        mask = preds[..., 4] > self.obj_threshold

        preds = [p[mask[idx]] for idx, p in enumerate(preds)][0]
        boxes = preds[..., :4].numpy()
        scores =  preds[..., 4].numpy()
        classes = preds[..., 5].numpy().astype(np.int64)

        return boxes, classes, scores
    

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
    
    def __init__(self, queue, cfg:None, onnx:True):
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
        co_helper = COCO_test_helper(enable_letter_box=True)
        boxes, scores, classes = out
        boxes = co_helper.get_real_box(boxes)

        for box, score, cl in zip(boxes, scores, classes):
            top, left, right, bottom = [int(_b) for _b in box]
            print("%s @ (%d %d %d %d) %.3f" % (CLASSES[cl], top, left, right, bottom, score))
            cv2.rectangle(frame, (top, left), (right, bottom), (255, 0, 0), 2)
            cv2.putText(frame, '{0} {1:.2f}'.format(CLASSES[cl], score),
                        (top, left - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)        
        cv2.imshow('Yolo_v10 Inference', frame)
        cv2.waitKey(1)