import unittest
import requests
from dotenv import dotenv_values

config = dotenv_values(".env")
ENDPOINT = 'http://127.0.0.1:5000'
# ENDPOINT = 'http://192.168.1.200:5000'
HEADERS = {"Authorization": f"Bearer {config['APP_TOKEN']}"}


class TestApi(unittest.TestCase):
    def test_home(self):
        resp = requests.get(ENDPOINT)
        self.assertIn('Objects detection', resp.text)

    def test_api(self):
        data = {'file': 42}
        resp = requests.post(ENDPOINT + '/predict/v1',
                             json=data,
                             headers=HEADERS)
        self.assertIn('boxes', resp.text)

    def test_file_upload(self):
        with open('data/bus.jpg', 'rb') as f:
            files = {'file': f}
            resp = requests.post(ENDPOINT + '/predict/v1',
                                 files=files,
                                 headers=HEADERS)

            print(resp.text)


if __name__ == '__main__':
    unittest.main()
