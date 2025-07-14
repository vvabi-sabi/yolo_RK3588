from pathlib import Path
from multiprocessing import Process, Queue
from rknnlite.api import RKNNLite

ROOT = Path(__file__).parent.parent.parent.absolute()
MODELS_PATH = str(ROOT) + "/yolo_RK3588/models/"


RKNNModelNames = {'yolov8':'yolov8n_batch.rknn',
                 }

def get_model_names(model_list):
    path_list = []
    for model in model_list:
        path_list.append(RKNNModelNames.get(model))
    return path_list

def get_model_path(model_names):
    return MODELS_PATH+model_names[0]



class RKNNModelLoader():
    
    def __init__(self):
        self.verbose = False
        self.verbose_file = 'verbose.txt'
    
    @staticmethod
    def load_weights(core, model):
        model = get_model_path(model)
        rknnlite = RKNNLite(verbose=False)
        print(f"Export rknn model - {model}")
        ret = rknnlite.load_rknn(model)
        if ret != 0:
            print(f'Export {model} model failed!')
            return ret
        print('Init runtime environment')
        ret = rknnlite.init_runtime(async_mode=False,
                                    core_mask = core
                                    )
        if ret != 0:
            print('Init runtime environment failed!')
            return ret
        print(f'{model} is loaded')
        return rknnlite


class Inference(Process):
    
    def __init__(self, input, rknnlite):
        super().__init__(group=None, target=None, name=None, args=(), kwargs={}, daemon=True)
        self.input = input
        self.q_out = Queue(maxsize=3)
        self._rknnlite = rknnlite
    
    def run(self):
        while True:
            frame = self.input.get()
            inference_result = self._rknnlite.inference(inputs=[frame])
            self.q_out.put((frame, inference_result))


class Net():
    
    def __init__(self, model_name, cores, q_input):
        self._model_name = get_model_names([model_name])
        self._rknnlite = RKNNModelLoader.load_weights(cores, self._model_name)
        self.inference = Inference(q_input, self._rknnlite)
