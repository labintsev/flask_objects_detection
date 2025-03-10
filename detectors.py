import cv2
import numpy as np
import time

from yolov5 import post_process, nms_boxes

ANCHORS = 'data/anchors_yolov5.txt'
DEVICE_NAME = 'RK3588'
DEVICE_COMPATIBLE_NODE = '/proc/device-tree/compatible'

ONNX_MODEL = 'models/yolov5s_relu.onnx'


def rknn_builder():
    # return DetectorRknnYolo11('./models/yolo11n_rknn/yolo11n.rknn')
    return DetectorRknnYolo5('./models/yolov5s_relu.rknn')


def onnx_builder():
    return DetectorOnnx(ONNX_MODEL)


class DetectorRknnYolo5:
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

    def predict(self, image_file):
        image = self.preprocess(image_file)
        print('--> Running model')
        t0 = time.time()
        outputs = self.rknn_lite.inference(inputs=[image])
        print('output shape: ', outputs[0].shape)
        boxes, classes, scores = post_process(outputs, self.anchors)
        print('Inference take: ', time.time() - t0, 'ms')
        return str(boxes), str(classes), str(scores)
    
    def preprocess(self, image):
        file_bytes = np.frombuffer(image.read(), dtype=np.uint8)
        img_data_ndarray = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img_data_ndarray, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 640))
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
        return str(boxes.astype(int)), str(classes.astype(int)), str(scores)
    
    def release(self):
        del(self.onnx_session)


class DetectorOnnxSliced(DetectorOnnx):
    def __init__(self, model_path):
        super().__init__(model_path)

    def preprocess(self, image):
        file_bytes = np.frombuffer(image.read(), dtype=np.uint8)
        img_data_ndarray = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img_data_ndarray, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (1920, 1920))
        img = np.expand_dims(img, 0)
        return img
    
    def predict(self, image):
        img = self.preprocess(image)
        return predict_sliced(img, self.onnx_session)


class DetectorRknnYolo11:
    def __init__(self, w):
        from rknnlite.api import RKNNLite
        self.model = RKNNLite()
        self.model.load_rknn(w)
        self.model.init_runtime()
    
    def preprocess(self, image_file):
        file_bytes = np.frombuffer(image_file.read(), dtype=np.uint8)
        img_ndarray = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img_ndarray, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 640))
        img = np.expand_dims(img, 0)
        return img
    
    def predict(self, image_file):
        image = self.preprocess(image_file)
        result = self.model.inference(inputs=[image])
        print('Outputs shape: ', result[0].shape)
        boxes = result[0][0, :4, :]
        boxes = np.swapaxes(boxes, 0, 1)
        scores = result[0][0, 4, :]
        top_idxs = np.argsort(scores)[:10]
        top_scores = np.sort(scores)[::-1][:10]
        top_boxes = boxes[top_idxs]
        xyxy = np.copy(top_boxes)
        xyxy[:, 0] = top_boxes[:, 0] - top_boxes[:, 2] / 2  # top left x
        xyxy[:, 1] = top_boxes[:, 1] - top_boxes[:, 3] / 2  # top left y
        xyxy[:, 2] = top_boxes[:, 0] + top_boxes[:, 2] / 2  # bottom right x
        xyxy[:, 3] = top_boxes[:, 1] + top_boxes[:, 3] / 2  # bottom right y
        print(top_scores)
        return str(xyxy.astype(int)), [1]*10, str(top_scores)
    

def slice_image(image, window_size):
    """Generate sliding windows for the image."""
    for y in range(3):
        for x in range(3):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])


def predict_sliced(img, detector):
    window_size = (300, 300)  # Define the size of the window

    all_outputs = []

    # Inference
    print('--> Running model')
    t0 = time.time()
    for (x, y, window) in slice_image(img[0], window_size):
        window = np.expand_dims(window, 0)
        outputs = detector.predict(img)
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
