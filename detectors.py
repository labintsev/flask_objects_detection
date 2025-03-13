import cv2
import numpy as np
import time
import logging
from abc import ABC, abstractmethod

from yolov5 import post_process

ANCHORS = "data/anchors_yolov5.txt"
log = logging.getLogger('detector')

class Detector(ABC):
    @abstractmethod
    def preprocess(self, image_file):
        """
        Preprocess the input image file for model inference.

        Args:
            image_file (file-like object): The input image file to be preprocessed.

        Returns:
            numpy.ndarray: The preprocessed image ready for model inference.
        """
        pass

    @abstractmethod
    def predict(self, image_file):
        """
        Perform object detection on the input image file.

        Args:
            image_file (file-like object): The input image file to be processed.

        Returns:
            tuple: A tuple containing:
                - str: The bounding boxes of detected objects. XYXY format.
                - str: A list of class labels for the detected objects.
                - str: The confidence scores of the detected objects.
        """
        pass

    @abstractmethod
    def postprocess(self, outputs):
        """
        Postprocess the model outputs to extract bounding boxes, class labels, and confidence scores.

        Args:
            outputs (list): The raw outputs from the model inference. 
                            Any boxes format, as model specific.

        Returns:
            tuple: A tuple containing:
                - numpy.ndarray: The bounding boxes of detected objects. XYXY format.
                - numpy.ndarray: The class labels for the detected objects.
                - numpy.ndarray: The confidence scores of the detected objects.
        """
        pass

    @abstractmethod
    def release(self):
        """
        Release any resources held by the detector.
        """
        pass


def rknn_builder():
    # return DetectorRknnYolo11('./models/yolo11n_rknn/yolo11n.rknn')
    # return DetectorRknnYolo5("./models/yolov5s_relu.rknn")
    return DetectorRknnYolo5Sliced("./models/yolov5s_relu.rknn")


def onnx_builder():
    # return DetectorOnnx(ONNX_MODEL)
    return DetectorOnnxSliced("models/yolov5s_relu.onnx")


class DetectorRknnYolo5(Detector):
    def __init__(self, model_path):
        from rknnlite.api import RKNNLite

        self.rknn_lite = RKNNLite()
        self.anchors = np.loadtxt(ANCHORS).reshape(3, -1, 2).tolist()
        log.info("--> Load RKNN model")
        ret = self.rknn_lite.load_rknn(model_path)
        if ret != 0:
            log.error("Load RKNN model failed")
            exit(ret)
        log.info("--> Init runtime environment")
        ret = self.rknn_lite.init_runtime()
        if ret != 0:
            log.error("Init runtime environment failed")
            exit(ret)


    def predict(self, image_file):
        image = self.preprocess(image_file)
        outputs = self.rknn_lite.inference(inputs=[image])
        boxes, classes, scores = post_process(outputs, self.anchors)
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


class DetectorRknnYolo5Sliced(Detector):
    def __init__(self, model_path):
        from rknnlite.api import RKNNLite

        self.rknn_lite = RKNNLite()
        self.anchors = np.loadtxt(ANCHORS).reshape(3, -1, 2).tolist()
        log.info("--> Load RKNN model")
        ret = self.rknn_lite.load_rknn(model_path)
        if ret != 0:
            log.error("Load RKNN model failed")
            exit(ret)
        log.info("--> Init runtime environment")
        ret = self.rknn_lite.init_runtime()
        if ret != 0:
            log.error("Init runtime environment failed")
            exit(ret)
        self.window_size = (640, 640)
        self.k_x = 1.0
        self.k_y = 1.0
    
    def preprocess(self, image):
        file_bytes = np.frombuffer(image.read(), dtype=np.uint8)
        img_data_ndarray = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img_data_ndarray, cv2.COLOR_BGR2RGB)
        return self.crops_generator(img)
    
    def crops_generator(self, img):
        img_width, img_height = img.shape[1], img.shape[0]
        crops_x = img_width // self.window_size[0]
        crops_y = img_height // self.window_size[1]
        self.k_x = img_width / (self.window_size[0] * crops_x)
        self.k_y = img_height / (self.window_size[1] * crops_y)
        img = cv2.resize(img, (crops_x * self.window_size[0], crops_y * self.window_size[1]))
        for y in range(crops_y):
            for x in range(crops_x):
                crop = img[y * self.window_size[1] : (y + 1) * self.window_size[1], 
                        x * self.window_size[0] : (x + 1) * self.window_size[0]]
                crop = np.expand_dims(crop, 0)
                yield x, y, crop

    def predict(self, image_file):
        img_crops = self.preprocess(image_file)
        all_boxes, all_classes, all_scores = [], [], []
        for x, y, crop in img_crops:
            outputs = self.rknn_lite.inference(inputs=[crop])
            boxes, classes, scores = post_process(outputs, self.anchors)
            if boxes is not None:
                for box in boxes:
                    # Adjust the coordinates based on the window position
                    adjusted_boxes = [
                        (box[0] + x * 640) * self.k_x,
                        (box[1] + y * 640) * self.k_y,
                        (box[2] + x * 640) * self.k_x,
                        (box[3] + y * 640) * self.k_y,
                    ]
                    all_boxes.append(adjusted_boxes)
                    all_classes.extend(classes)
                    all_scores.extend(scores)
        all_boxes = np.array(all_boxes, dtype=int)
        all_classes = np.array(all_classes, dtype=int)
        all_scores = np.array(all_scores)
        return str(all_boxes), str(all_classes), str(all_scores)

    def postprocess(self, outputs):
        return post_process(outputs, self.anchors)
    
    def release(self):
        self.rknn_lite.release()


class DetectorOnnx(Detector):
    def __init__(self, model_path):
        import onnxruntime

        self.model_path = model_path
        self.anchors = np.loadtxt(ANCHORS).reshape(3, -1, 2).tolist()
        self.onnx_session = onnxruntime.InferenceSession(model_path)
        log.info("ONNX inputs: ", self.onnx_session.get_inputs()[0])

    def predict(self, image):
        img = self.preprocess(image)
        raw_output = self.onnx_session.run(None, {"images": img})
        output = self.postprocess(raw_output)
        return output

    def preprocess(self, image):
        file_bytes = np.frombuffer(image.read(), dtype=np.uint8)
        img_data_ndarray = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img_data_ndarray, cv2.COLOR_BGR2RGB)
        input_data = img.transpose((2, 0, 1))
        input_data = np.expand_dims(input_data, 0).astype(np.float32)
        input_data = input_data / 255.0
        return input_data

    def postprocess(self, outputs):
        boxes, classes, scores = post_process(outputs, self.anchors)
        return str(boxes.astype(int)), str(classes.astype(int)), str(scores)

    def release(self):
        del self.onnx_session


class DetectorOnnxSliced(Detector):
    def __init__(self, model_path):
        import onnxruntime

        self.model_path = model_path
        self.anchors = np.loadtxt(ANCHORS).reshape(3, -1, 2).tolist()
        self.onnx_session = onnxruntime.InferenceSession(model_path)
        log.info("ONNX inputs: ", self.onnx_session.get_inputs()[0])
        self.window_size = (640, 640)
        self.k_x = 1.0
        self.k_y = 1.0
    
    def preprocess(self, image):
        file_bytes = np.frombuffer(image.read(), dtype=np.uint8)
        img_data_ndarray = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img_data_ndarray, cv2.COLOR_BGR2RGB)
        return self.crops_generator(img)

    def predict(self, image_file):
        img_crops = self.preprocess(image_file)
        all_boxes, all_classes, all_scores = [], [], []
        for x, y, crop in img_crops:
            raw_output = self.onnx_session.run(None, {"images": crop})
            boxes, classes, scores = post_process(raw_output, self.anchors)
            if boxes is not None:
                for box in boxes:
                    # Adjust the coordinates based on the window position
                    adjusted_boxes = [
                        (box[0] + x * 640) * self.k_x,
                        (box[1] + y * 640) * self.k_y,
                        (box[2] + x * 640) * self.k_x,
                        (box[3] + y * 640) * self.k_y,
                    ]
                    all_boxes.append(adjusted_boxes)
                    all_classes.extend(classes)
                    all_scores.extend(scores)

        all_boxes = np.array(all_boxes, dtype=int)
        all_classes = np.array(all_classes, dtype=int)
        all_scores = np.array(all_scores)
        return str(all_boxes), str(all_classes), str(all_scores)

    def crops_generator(self, img):
        img_width, img_height = img.shape[1], img.shape[0]
        crops_x = img_width // self.window_size[0]
        crops_y = img_height // self.window_size[1]
        self.k_x = img_width / (self.window_size[0] * crops_x)
        self.k_y = img_height / (self.window_size[1] * crops_y)
        img = cv2.resize(img, (crops_x * self.window_size[0], crops_y * self.window_size[1]))
        for y in range(crops_y):
            for x in range(crops_x):
                crop = img[y * self.window_size[1] : (y + 1) * self.window_size[1], 
                        x * self.window_size[0] : (x + 1) * self.window_size[0]]
                crop = crop.transpose((2, 0, 1))
                crop = np.expand_dims(crop, 0).astype(np.float32)
                crop = crop / 255.0
                yield x, y, crop

    def postprocess(self, outputs):
        return post_process(outputs, self.anchors)

    def release(self):
        del self.onnx_session


class DetectorRknnYolo11(Detector):
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
        return str(xyxy.astype(int)), [1] * 10, str(top_scores)
