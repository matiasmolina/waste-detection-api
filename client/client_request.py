# Matias D. Molina - molinamatiasd@gmail.com - linkedin.com/in/matiasmolina

import json
import base64
from PIL import Image
import io
import sys
import numpy as np
import requests

from read_the_output import image_bytes_to_numpy

def show_response(response):
    content = response.content
    status_code = response.status_code

    print('>>', 'status code: ', status_code)
    content = json.loads(response.text)

    for k in content.keys():
        if k not in  ['data', 'errors']:
            print('>>', k.title(), content[k])
        else:
            if content[k] is not None:
                print('>>', k.title()+'?', len(content[k]) > 0)
            else:
                print('>>', k.title()+'?', False)

    if 'data' in content.keys() and content['data'] is not None:
        for k in content['data'].keys():
            if k not in ['input_image', 'output_image']:
                print('>>', k, content['data'][k])
            else:
                if content['data'][k] is not None:
                    im = base64.b64decode(content['data']['input_image'])
                    print('>> data', k.title(), image_bytes_to_numpy(im).shape)
                else:
                    print('>> data', k.title(), content['data'][k])
    if 'errors' in content.keys() and content['errors'] is not None:
        print(content['errors'])

# Here are some examples for a quick verification.
# 1. Success case with a positive detection
print('** Success Case **')
url = "http://localhost:5000/predict"
data = {
    "latitude": -34.82929722222222,
    "longitude": -58.40813611111111,
    "zoom": 18,
    "quantile": 0.94
}

response = requests.post(url, json=data)
show_response(response)
print('')

#2.  Success case with Negative detection
print('** Success Case **')
url = "http://localhost:5000/predict"
data = {
    "latitude": 37.653770,
    "longitude": -7.54779,
    "zoom": 18,
    "quantile": 0.94
}

response = requests.post(url, json=data)
show_response(response)
print('')

# 3. Bad latitude case:
print('** Latitude Error Case **')
url = "http://localhost:5000/predict"
data = {
    "latitude": 48653770,
    "longitude": -7.54779,
    "zoom": 18,
    "quantile": 0.94
}

response = requests.post(url, json=data)
show_response(response)
print('')

# 4. Bad latitude case:
print('** Latitude Error Case 2**')
url = "http://localhost:5000/predict"
data = {
    "latitude": "lalala",
    "longitude": -7.54779,
    "zoom": 18,
    "quantile": 0.94
}

response = requests.post(url, json=data)
show_response(response)
print('')
