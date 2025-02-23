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

    def test_boxes_in_respones(self):
        with open('data/bus.jpg', 'rb') as f:
            files = {'image_file': f}
            resp = requests.post(ENDPOINT + '/predict',
                                    files=files,
                                    headers=HEADERS)
        # print(resp.text)
        self.assertIn('boxes', resp.text)



if __name__ == '__main__':
    unittest.main()
