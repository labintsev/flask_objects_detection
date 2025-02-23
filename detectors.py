import cv2
import numpy as np
import time

from yolov5 import post_process

ANCHORS = 'data/anchors_yolov5.txt'

RK3588_RKNN_MODEL = 'models/yolov5s_relu.rknn'
DEVICE_NAME = 'RK3588'
DEVICE_COMPATIBLE_NODE = '/proc/device-tree/compatible'

ONNX_MODEL = 'models/yolov5s_relu.onnx'

class DetectorRknnLite:
    def __init__(self, model_path):
        from rknnlite.api import RKNNLite
        self.rknn_lite = RKNNLite()
        self.anchors = np.loadtxt(ANCHORS).reshape(3, -1, 2).tolist()
        print('--> Load RKNN model')
        ret = self.rknn_lite.load_rknn(model_path)
        if ret != 0:
            print('Load RKNN model failed')
            exit(ret)
        print('done')
        print('--> Init runtime environment')
        ret = self.rknn_lite.init_runtime()
        if ret != 0:
            print('Init runtime environment failed')
            exit(ret)
        print('done')

    def predict(self, image):
        return predict_v1(image, self.rknn_lite)
    
    def preprocess(self, image):
        file_bytes = np.frombuffer(image.read(), dtype=np.uint8)
        img_data_ndarray = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img_data_ndarray, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img, 0)
        return img
    
    def postprocess(self, outputs):
        boxes, classes, scores = post_process(outputs, self.anchors)
        return boxes, classes, scores
    
    def release(self):
        self.rknn_lite.release()


class DetectorOnnx:
    def __init__(self, model_path):
        import onnxruntime
        self.model_path = model_path
        self.anchors = np.loadtxt(ANCHORS).reshape(3, -1, 2).tolist()
        self.onnx_session = onnxruntime.InferenceSession(model_path)
        print('ONNX inputs: ', self.onnx_session.get_inputs()[0])
    
    def predict(self, image):
        img = self.preprocess(image)
        raw_output = self.onnx_session.run(None, {'images': img})
        output = self.postprocess(raw_output)
        print(output)
        return output
    
    def preprocess(self, image):
        file_bytes = np.frombuffer(image.read(), dtype=np.uint8)
        img_data_ndarray = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img_data_ndarray, cv2.COLOR_BGR2RGB)
        input_data = img.transpose((2,0,1))
        input_data = np.expand_dims(input_data, 0).astype(np.float32)
        input_data = input_data/255.
        return input_data
    
    def postprocess(self, outputs):
        boxes, classes, scores = post_process(outputs, self.anchors)
        return str(boxes), str(classes), str(scores)
    
    def release(self):
        del(self.onnx_session)


def rknn_builder():
    return DetectorRknnLite(RK3588_RKNN_MODEL)

def onnx_builder():
    return DetectorOnnx(ONNX_MODEL)


def predict_v1(image, rknn_lite):
    file_bytes = np.frombuffer(image.read(), dtype=np.uint8)
    print(file_bytes)
    img_data_ndarray = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img_data_ndarray, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, 0)

    with open(ANCHORS, 'r') as f:
        values = [float(_v) for _v in f.readlines()]
        anchors = np.array(values).reshape(3,-1,2).tolist()
        print("use anchors from '{}', which is {}".format(ANCHORS, anchors))
    
    # Inference
    print('--> Running model')
    t0 = time.time()
    outputs = rknn_lite.inference(inputs=[img])
    boxes, classes, scores = post_process(outputs, anchors)

    print('Inference take: ', time.time() - t0, 'ms')
    print(boxes, classes, scores)
    return str(boxes), str(classes), str(scores)



def slice_image(image, window_size):
    """Generate sliding windows for the image."""
    for y in range(3):
        for x in range(3):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])


def predict_sliced(img, rknn_lite):
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
