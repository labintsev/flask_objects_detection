"""Objects detection service"""
import os
import time
import atexit

from dotenv import dotenv_values
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
from flask_cors import CORS
from flask_httpauth import HTTPTokenAuth

import cv2
import numpy as np

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
CORS(app)

config = dotenv_values(".env")
auth = HTTPTokenAuth(scheme='Bearer')
tokens = {
    config['APP_TOKEN']: "user1",
}


@atexit.register
def release_rknn():
    print('Release rknn device')


def dummy_model_predict(img):
    return [[0., 0., 0.1, 0.1]]


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_image(input_data):
    return cv2.imread('data/bus.jpg')


def predict(input_data):
    image = get_image(input_data)

    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, 0)

    # Inference
    print('--> Running model')
    t0 = time.time()
    outputs = dummy_model_predict(img)
    print('Inference take: ', time.time() - t0, 'ms')

    return outputs


@auth.verify_token
def verify_token(token):
    if token in tokens:
        return tokens[token]


@app.route("/")
def home():
    return '<h1>Objects detection service.</h1> Use /predict endpoint'


@app.route("/predict", methods=['POST'])
@auth.login_required
def predict_web_serve():
    """OD service"""
    boxes = []

    # check if the post request has the file part
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    # If the user does not select a file, the browser submits an
    # empty file without a filename.
    if file.filename == '':
        flash('No selected file')
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        boxes = predict(file)
    return {'boxes': boxes}


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)


