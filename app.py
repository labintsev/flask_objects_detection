"""House price prediction service"""
import argparse
import time
import atexit
import os

from dotenv import dotenv_values
from flask import Flask, flash, request, redirect
from werkzeug.utils import secure_filename
from flask_cors import CORS
from flask_httpauth import HTTPTokenAuth

import cv2
import numpy as np

from yolov5 import post_process
from predict_utils import init_rknn_lite, predict

RK3588_RKNN_MODEL = 'yolov5s_relu.rknn'
DEVICE_NAME = 'RK3588'
DEVICE_COMPATIBLE_NODE = '/proc/device-tree/compatible'

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "super secret key rknn"

CORS(app)

config = dotenv_values(".env")
auth = HTTPTokenAuth(scheme='Bearer')
tokens = { config['APP_TOKEN']: "user1", }

app.config['RKNN_LITE'] = init_rknn_lite(RK3588_RKNN_MODEL)
ANCHORS = 'anchors_yolov5.txt'

@atexit.register
def release_rknn():
    app.config['RKNN_LITE'].release()
    print('Release rknn device')


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def predict_v0(image):
    rknn_lite = app.config['RKNN_LITE']
    file_bytes = np.frombuffer(image.read(), dtype=np.uint8)
    print(file_bytes)
    img_data_ndarray = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img_data_ndarray, cv2.COLOR_BGR2RGB)
    img.resize((900, 900, 3))
    img = np.expand_dims(img, 0)

    # Inference
    print('--> Running model')
    t0 = time.time()
    outputs = predict(img, rknn_lite)
    print('Inference take: ', time.time() - t0, 'ms')

    return str(outputs)


def predict_v1(image):
    rknn_lite = app.config['RKNN_LITE']
    file_bytes = np.frombuffer(image.read(), dtype=np.uint8)
    print(file_bytes)
    img_data_ndarray = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img_data_ndarray, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, 0)

    # load anchor
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


@auth.verify_token
def verify_token(token):
    if token in tokens:
        return tokens[token]


@app.route("/")
def home():
    return '<h1>Objects detection service.</h1> Use /predict endpoint'


@app.route("/predict", methods=['POST', 'GET'])
@auth.login_required
def predict_web_serve():
    """OD service"""
    if request.method == 'POST':
        boxes = []

        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        image = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if image.filename == '':
            flash('No selected file')
        if image and allowed_file(image.filename):
            filename = secure_filename(image.filename)
            boxes = predict(image)
            # image.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            
        return {'boxes': boxes}
    else:
        return {'boxes': '[]'}



@app.route("/predict/v1", methods=['POST', 'GET'])
@auth.login_required
def predict_yolov5():
    """Objects detection service v1"""
    print(request)

    boxes = []

    # check if the post request has the file part
    if 'image_file' not in request.files:
        flash('No file part')
        return {'error': "No file in the request"}
    image = request.files['image_file']
    # If the user does not select a file, the browser submits an
    # empty file without a filename.
    if image.filename == '':
        flash('No selected file')
    if image and allowed_file(image.filename):
        filename = secure_filename(image.filename)
        boxes, classes, scores = predict_v1(image)
        image.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        
    return {'boxes': boxes, 'classes': classes, 'scores': scores}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--engine', nargs='+',
                        help='Set inference engine: rknn or onnx, default is onnx',
                        default='onnx')
    args = parser.parse_args()
    app.run(host='0.0.0.0', debug=True)


