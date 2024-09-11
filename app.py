# Matias D. Molina - molinamatiasd@gmail.com - linkedin.com/in/matiasmolina

from flask import Flask, request, jsonify
import pickle
import base64
import io

import torch
import torchvision.transforms as transforms
from PIL import Image

# Model libs
from classification_model.downloader import ImageDownloader
from classification_model.query import ModelQuery
from classification_model.image_utils import overlay_image_mask

# Argument validators
from validators import (is_valid_latitude, is_valid_longitude,
                       is_valid_zoom, is_valid_quantile, check_arguments)
# Service settings
from settings import (secret_key, model_path,
                      default_zoom, default_pixel_quantile,
                      positive_message, negative_message)

from settings import port_number

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Internal normalizer
transform = transforms.Compose([
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

### Utils
def numpy_to_bytes(arr):
    image = Image.fromarray(arr.astype('uint8'))

    # PIL Image to byte buffer
    byte_buffer = io.BytesIO()
    image.save(byte_buffer, format='PNG')
    byte_buffer.seek(0)

    # Get the byte data
    byte_data = byte_buffer.read()

    return byte_data


### Methods
@app.route('/predict', methods=['POST'])
def predict():
    
    if request.is_json:
        data = request.json
    else:
        data = request.form

    # Value checking.
    if 'zoom' in data:
        zoom = data['zoom']
    else:
        zoom = default_zoom

    if 'quantile' in data:
        quantile = data['quantile']
    else:
        quantile = default_pixel_quantile

    if 'latitude' in data:
        lat = data['latitude']
    else:
        return jsonify({'status': 'error',
                        'message': 'Latitude is a mandatory argument.',
                        'errors': None,
                        'data': None,
                }), 400

    if 'longitude' in data:
        lon = data['longitude']
    else:
        return jsonify({'status': 'error',
                        'message': 'Longitude is a mandatory argument.',
                        'errors': None,
                        'data': None,
                }), 400 

    err_msg = check_arguments(zoom, quantile, lat, lon)

    if len(err_msg) > 0:
        return jsonify({'status': 'error',
                        'message': 'Errors encountered in argument values.',
                        'errors': err_msg,
                        'data': None,
                        }), 400
 
    # Loading Model
    try:
        model = torch.load(model_path, map_location=torch.device(device))
    except Exception:
        return jsonify({'status': 'error',
                        'message': 'Error when loading the service.',
                        'errors': None,
                        'data': None,
                        }), 500

    # Image downloading
    location = (lat, lon)
    try:
        downloader = ImageDownloader(location=location,
                                     zoom=zoom,
                                     secret_key=secret_key)
        image = downloader.request()
    except Exception:
        return jsonify({'status': 'error',
                        'message': 'Errors when downloading the image from satellite service.',
                        'errors': None,
                        'data': None,
                        }), 500

    # Query
    try:
        query = ModelQuery(model=model,
                       quantile=quantile,
                       device=device,
                       transform=transform)

        pred, score, cam = query.make_query(image.transpose(2,0,1))
    except Exception as e:
        return jsonify({'status': 'error',
                        'message': 'Error during query process.',
                        'errors': None,
                        'data': None,
                        }), 500

    # Process the response
    final_image = None
    if cam is not None:
        final_image = overlay_image_mask(image, cam, 0.5, 'jet')

    if final_image is not None:
        # Convert to bytes and encode it.
        final_image = numpy_to_bytes(final_image)
        final_image = base64.b64encode(final_image).decode('utf-8')

    # Convert original image:
    image = numpy_to_bytes(image)
    image = base64.b64encode(image).decode('utf-8')

    if pred:
        msg = positive_message
    else:
        msg = negative_message

    response = {'status': 'success',
                'message': msg,
                'data': {'prediction': pred,
                         'score': score,
                         'output_image': final_image,
                         'input_image': image,
                         'location': location,
                         'zoom': zoom,
                         'quantile': quantile
                        },
                'errors': None,
                }
    return jsonify(response), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port_number)