import os
import numpy as np
import cv2
from pycocotools.coco import COCO
from multiprocessing import Process


class Camera(Process):
    """This class represents a camera process that captures frames from a video source 
    and performs various operations on the frames (preprocess).

    Attributes
    ----------
    net_size : tuple
        The size of the neural network input.
    queue : Queue
        The queue to put the processed frames into.
    source : int, str
        The video source (also path to file.mp4) to capture frames from.
    frames : generator
        A generator object that yields frames from the video capture.

    Methods
    -------
    get_frame(None)
        Returns the next frame from the frames generator.
    resize_frame(frame, net_size)
        Resizes the given frame using OpenCV's resize function.
    crop_frame(frame, net_size)
        Crops the given frame based on net_size.
    run(None) 
        Iterates over the frames generator, processes each frame, and puts it into the queue.

    """

    def __init__(self, source: int, queue, onnx=True, gt_queue=None):
        """
        Parameters
        ----------
        source : int, str
            The video source.
        queue : Queue
            The queue in which processed frames are placed. Then these frames will be fed 
            to the input of the neural network.
        onnx : bool, optional
            Whether to use ONNX model for postprocessing. Defaults to True.
        """
        super().__init__(group=None, target=None, name=None, args=(), kwargs={}, daemon=True)
        INPUT_SIZE = (550 if onnx else 544)
        self.net_size = (INPUT_SIZE, INPUT_SIZE) 
        self._queue = queue
        self._gt_queue = gt_queue
        self.source = source

    @property
    def frames(self):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            raise SystemExit("Bad source")
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    raise SystemExit("Camera stopped!")
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                yield frame
        except Exception as e:
            print(f"Stop recording loop. Exception {e}")
        finally:
            cap.release()
    
    @staticmethod
    def letterbox(
        im,
        new_shape=(640, 640),
        color=(114, 114, 114),
        auto=True,
        scaleup=True,
        stride=32,
    ) -> tuple[np.ndarray, float, tuple[float, float]]:

        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        print('shape[0], shape[1]', shape[0], shape[1])
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(
            im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
        )  # add border
        return im
    
    def get_frame(self):
        """It yields the frame, making it available for further processing outside the function.
        """
        return next(self.frames)

    def resize_frame(self, frame, net_size=(1920, 1080)):
        '''
            useless if frame.shape == (1080, 1920)
        '''
        frame_size = frame.shape[:2]
        interpolation = cv2.INTER_CUBIC if any(x < y for x, y in zip(frame_size, net_size)) else cv2.INTER_AREA
        return cv2.resize(frame, net_size, interpolation=interpolation)
    
    def crop_frame(self, frame):
        crop1 = frame[:540, 0:640, :]
        crop2 = frame[:540, 640:1280, :]
        crop3 = frame[:540, 1280:, :]
        crop4 = frame[540:, 0:640, :]
        crop5 = frame[540:, 640:1280, :]
        crop6 = frame[540:, 1280:, :]
        return [crop1, crop2, crop3, crop4, crop5, crop6]

    def run(self):
        '''When processing a raw frame, there are two methods to choose from:
        resize_frame or crop_frame.
        '''
        for raw_frame in self.frames:
            frame = self.resize_frame(raw_frame) # useless function
            frames = self.crop_frame(frame)
            frames = [self.letterbox(crop, new_shape=(640, 640), auto=False) for crop in frames]
            frames = [np.expand_dims(crop, axis=0) for crop in frames]
            if (not self._queue.empty() and type(self.source) == int):
                continue
            self._queue.put((np.concat(frames)))


COCO_LABEL_MAP = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8,
                  9: 9, 10: 10, 11: 11, 13: 12, 14: 13, 15: 14, 16: 15, 17: 16,
                  18: 17, 19: 18, 20: 19, 21: 20, 22: 21, 23: 22, 24: 23, 25: 24,
                  27: 25, 28: 26, 31: 27, 32: 28, 33: 29, 34: 30, 35: 31, 36: 32,
                  37: 33, 38: 34, 39: 35, 40: 36, 41: 37, 42: 38, 43: 39, 44: 40,
                  46: 41, 47: 42, 48: 43, 49: 44, 50: 45, 51: 46, 52: 47, 53: 48,
                  54: 49, 55: 50, 56: 51, 57: 52, 58: 53, 59: 54, 60: 55, 61: 56,
                  62: 57, 63: 58, 64: 59, 65: 60, 67: 61, 70: 62, 72: 63, 73: 64,
                  74: 65, 75: 66, 76: 67, 77: 68, 78: 69, 79: 70, 80: 71, 81: 72,
                  82: 73, 84: 74, 85: 75, 86: 76, 87: 77, 88: 78, 89: 79, 90: 80}
