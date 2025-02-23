import cv2
import numpy as np
import time
from rknnlite.api import RKNNLite


def init_rknn_lite(model_path):
    rknn_lite = RKNNLite()
    # Load RKNN model
    print('--> Load RKNN model')
    ret = rknn_lite.load_rknn(model_path)
    if ret != 0:
        print('Load RKNN model failed')
        exit(ret)
    print('done')

    # Init runtime environment
    print('--> Init runtime environment')
    ret = rknn_lite.init_runtime()
    if ret != 0:
        print('Init runtime environment failed')
        exit(ret)
    print('done')
    return rknn_lite


def slice_image(image, window_size):
    """Generate sliding windows for the image."""
    for y in range(3):
        for x in range(3):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])


def predict(img, rknn_lite):
    window_size = (300, 300)  # Define the size of the window

    all_outputs = []

    # Inference
    print('--> Running model')
    t0 = time.time()
    for (x, y, window) in slice_image(img[0], window_size):
        window = np.expand_dims(window, 0)
        outputs = rknn_lite.inference(inputs=[img])
        for output in outputs:
            # Adjust the coordinates based on the window position
            adjusted_output = [
                output[0] + x / img.shape[2],
                output[1] + y / img.shape[1],
                output[2],
                output[3]
            ]
            all_outputs.append(adjusted_output)
    print('Inference take: ', time.time() - t0, 'ms')

    return all_outputs
