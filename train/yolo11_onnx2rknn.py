import sys
from rknn.api import RKNN

model_name = 'yolo11n'

DEFAULT_RKNN_PATH = f'./models/{model_name}.rknn'
DEFAULT_QUANT = False


if __name__ == '__main__':
    model_path, platform, do_quant, output_path = f"../models/{model_name}.onnx", "rk3588", False, DEFAULT_RKNN_PATH

    # Create RKNN object
    rknn = RKNN(verbose=True)

    # Pre-process config
    print('--> Config model')
    rknn.config(mean_values=[[0, 0, 0]],
                std_values=[[255, 255, 255]],
                target_platform=platform)
    print('done')

    # Load model
    print('--> Loading model')
    ret = rknn.load_onnx(model=model_path)
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=False)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    # Export rknn model
    print('--> Export rknn model')
    ret = rknn.export_rknn(output_path)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')

    # Release
    rknn.release()
