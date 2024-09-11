# Matias D. Molina - molinamatiasd@gmail.com - linkedin.com/in/matiasmolina

import json
import base64
from PIL import Image
import io
import sys
import numpy as np
import requests 

def image_bytes_to_numpy(image_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(image_bytes))
    image = image.convert('RGB')
    image = np.array(image)
    return image

if __name__ == '__main__':
## Example:
# 1. As Client, call the following
#  curl -X POST -H "Content-Type: application/json" -d '{"latitude":-34.8292972 , "longitude":58.4081361, "quantile":0.94, "zoom":18}' http://localhost:5000/predict > output.txt
#
# 2. if it works, run this scripts as
#    read_the_output.py output.txt output
# 3. Check the standard output messages
#    and the images saved as output_image_{in,out}.png
##
    path_in = sys.argv[1]
    path_out = sys.argv[2]

    f = open(path_in)
    content = f.read()

    cnt = json.loads(content)
    pred = cnt['data']['prediction']
    score = cnt['data']['score']
    output_image = cnt['data']['output_image']
    input_image = cnt['data']['input_image']

    print('RESPONSE:')
    print(f'Prediction: {pred}')
    print(f'Score: {score}')
    print(f'Message: {cnt["message"]}')

    if output_image is not None:
        img_bytes = base64.b64decode(output_image)
        arr = image_bytes_to_numpy(img_bytes)
        im = Image.fromarray(arr)
        im.save(path_out + 'image_out.png') # Check the saved image.
        print('Output image saved in ', path_out + '_image_out.png')
    else:
        print('No output image in data.')

    if input_image is not None:
        img_bytes = base64.b64decode(input_image)
        arr = image_bytes_to_numpy(img_bytes)
        im = Image.fromarray(arr)
        im.save(path_out + 'image_in.png') # Check the saved image.
        print('Input image saved in ', path_out + '_image_in.png')
    else:
        print('No input image in data.')