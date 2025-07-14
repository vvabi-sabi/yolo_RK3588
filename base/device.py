from rknnlite.api import RKNNLite

from .rknn_models import Net


class NeuroModule():

    _PROCESSES_NUMBER = 3

    def __init__(self, model_name, cores_list, q_input):
        self.net = Net(model_name, cores_list, q_input)

    def run_inference(self):
        self.net.inference.start()


class RK3588():

    _CORES = [RKNNLite.NPU_CORE_0,
              RKNNLite.NPU_CORE_1,
              RKNNLite.NPU_CORE_2,
              RKNNLite.NPU_CORE_AUTO,
              RKNNLite.NPU_CORE_0_1,
              RKNNLite.NPU_CORE_0_1_2
              ]

    def __init__(self, model_name, camera):
        self._camera = camera
        self._neuro = NeuroModule(model_name,
                                  self._CORES[5],
                                  self._camera._queue)
