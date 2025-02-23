import time

import cv2
import numpy as np
import platform
from rknnlite.api import RKNNLite

# device tree for RK356x/RK3576/RK3588
DEVICE_COMPATIBLE_NODE = '/proc/device-tree/compatible'


RK3588_RKNN_MODEL = 'yolov5s_relu.rknn'


if __name__ == '__main__':

    # Get device information
    host_name = 'RK3588'
    rknn_model = RK3588_RKNN_MODEL

    rknn_lite = RKNNLite()

    # Load RKNN model
    print('--> Load RKNN model')
    ret = rknn_lite.load_rknn(rknn_model)
    if ret != 0:
        print('Load RKNN model failed')
        exit(ret)
    print('done')

    ori_img = cv2.imread('data/bus.jpg')
    img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, 0)

    # Init runtime environment
    print('--> Init runtime environment')
    # Run on RK356x / RK3576 / RK3588 with Debian OS, do not need specify target.
    if host_name in ['RK3576', 'RK3588']:
        # For RK3576 / RK3588, specify which NPU core the model runs on through the core_mask parameter.
        ret = rknn_lite.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
    else:
        ret = rknn_lite.init_runtime()
    # ret = rknn_lite.init_runtime()
    if ret != 0:
        print('Init runtime environment failed')
        exit(ret)
    print('done')

    # Inference
    print('--> Running model')
    t0 = time.time()
    outputs = rknn_lite.inference(inputs=[img])
    print('Inference take: ', time.time() - t0, 'ms')
    # Show the classification results
    # print(outputs)
    print('done')

    rknn_lite.release()
