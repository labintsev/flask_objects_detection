"""Objects detection prediction service. 
Supported engines: onnx, rknn"""

import argparse
import atexit
import os

from dotenv import dotenv_values
from flask import Flask, flash, request
from werkzeug.utils import secure_filename
from flask_cors import CORS
from flask_httpauth import HTTPTokenAuth

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "super secret key rknn"

CORS(app)

config = dotenv_values(".env")
auth = HTTPTokenAuth(scheme='Bearer')
tokens = { config['APP_TOKEN']: "user1", }


@atexit.register
def release_detector_resources():
    try:
        app.config['DETECTOR'].release()
        print('Detector resources released')
    except:
        print('Error detector release')


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@auth.verify_token
def verify_token(token):
    if token in tokens:
        return tokens[token]


@app.route("/")
def home():
    return '<h1>Objects detection service.</h1> Use /predict endpoint'


@app.route("/predict", methods=['POST', 'GET'])
@auth.login_required
def predict_yolov5():
    """Objects detection service"""
    print(request)
    boxes = []
    if 'image_file' not in request.files:
        flash('No file part')
        return {'error': "No file in the request"}
    image = request.files['image_file']

    if image.filename == '':
        flash('No selected file')

    if image and allowed_file(image.filename):
        filename = secure_filename(image.filename)
        boxes, classes, scores = app.config['DETECTOR'].predict(image)
        image.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        
    return {'boxes': boxes, 'classes': classes, 'scores': scores}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--engine',
                        help='Set inference engine: rknn or onnx, default is onnx',
                        default='onnx')
    args = parser.parse_args()
    if args.engine == 'rknn':
        from detectors import rknn_builder
        app.config['DETECTOR'] = rknn_builder()
    else:
        from detectors import onnx_builder
        app.config['DETECTOR'] = onnx_builder()

    app.run(host='0.0.0.0', debug=True)
