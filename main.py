from multiprocessing import Queue

from utils import PostProcess, Visualizer
from base import Camera, RK3588


rknn_postprocess_cfg = {'img_size' : 640,
                        'conf_threshold' : 0.5,
                        'iou_threshold' : 0.5,
                    }


def run(device, visualizer, post_process):
    device._camera.start()
    device._neuro.run_inference()
    if post_process is not None:
        post_process.run()
        while True:
            frame, outputs = post_process.get_outputs()
            visualizer.show_results(frame, outputs)
            # gps_data = matcher.get_gps_data(frame, mask)

def main(source):
    """
    """
    queue_size = 5
    q_pre = Queue(maxsize=queue_size)
    model = 'yolov8'
    camera = Camera(source=source,
                    queue=q_pre)
    device = RK3588(model, camera)
    post_processes = PostProcess(queue=device._neuro.net.inference.q_out,
                                 cfg=rknn_postprocess_cfg)
    visualizer = Visualizer()
    try:
        run(device, visualizer, post_processes)
    except Exception as e:
        print("Main exception: {}".format(e))
        exit()


if __name__ == "__main__":
    camera_source = 11 # '/home/firefly/11.mp4'
    main(camera_source)
