# Flask Object Detection Service

This project provides an object detection service using YOLOv5 models. The service can run inference using either RKNN or ONNX models and is built using Flask.

## Project Structure

```
__pycache__/
data/
distr/
models/
uploads/
app.py
detectors.py
requirements.txt
test_api.py
test_rknn.py
```

- app.py: Main application file for the Flask server.
- detectors.py: Contains the detector classes for RKNN and ONNX models.
- requirements.txt: List of Python dependencies.
- test_api.py: Unit tests for the API.
- test_rknn.py: Script to test the RKNN model.
- data: Contains data files such as anchors and labels.
- models: Contains the model files.
- uploads: Directory for uploaded images.

## Setup

### Prerequisites

- Python 3.10 or higher
- `pip` package manager

### Installation

1. Clone the repository:

```sh
git clone https://github.com/labintsev/flask_objects_detection.git
cd flask_objects_detection
```

2. Create a virtual environment and activate it:

```sh
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

3. Install the dependencies:

```sh
pip install -r requirements.txt
```

4. Create a .env file in the root directory with the following content:

```
APP_TOKEN=your_secret_token
```

## Usage

### Running the Server

To start the Flask server, run:

```sh
python app.py
```

By default, the server will run on `http://0.0.0.0:5000`.

### API Endpoints

- `GET /`: Home endpoint. Returns a welcome message.
- `POST /predict`: Endpoint for object detection. Requires an image file and a valid token.

### Example Request

```sh
curl -X POST http://127.0.0.1:5000/predict \
     -H "Authorization: Bearer your_secret_token" \
     -F "image_file=@path_to_your_image.jpg"
```

## Testing

To run the unit tests, use:

```sh
python -m unittest test_api.py
```

## License

This project is licensed under the Apache License 2.0. See the LICENSE file for details.

## Acknowledgements

- [YOLOv5](https://github.com/ultralytics/yolov5)
- [Flask](https://flask.palletsprojects.com/)
- [RKNN Toolkit](https://github.com/rockchip-linux/rknn-toolkit)

## Contact

For any inquiries, please contact [andrej.labintsev@yandex.ru](mailto:andrej.labintsev@yandex.ru).