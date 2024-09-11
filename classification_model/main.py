# Matias D. Molina - molinamatiasd@gmail.com - linkedin.com/in/matiasmolina

from settings import secret_key

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
 
import numpy as np

from query import ModelQuery
from downloader import ImageDownloader
from image_utils import overlay_image_mask
import argparse

from matplotlib import pyplot as plt

def get_input_args():
    def parse_tuple(input_string):
        t = tuple(map(float, input_string.split(',')))
        if len(t) != 2:
            raise ValueError('Must be a pair of numbers.')
        return t

    parser = argparse.ArgumentParser()

    help_ = 'Pixel-wise score threshold.'
    parser.add_argument('--q', type=float, help=help_, default=0.94)

    help_ = 'Zoom level of satellite image.'
    parser.add_argument('--zoom', type=int, help=help_, default=18)

    help_ = 'Map location (latitude, longitude), eg. 12.345,56.789'
    parser.add_argument('--loc', type=parse_tuple, help=help_, required=True)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_input_args()
    zoom = args.zoom
    location = args.loc
    quantile = args.q


    # Internal normalizer
    transform = transforms.Compose([                                                                                    
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])        
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = './models/model_cls.pt'
    model = torch.load(model_path, map_location=torch.device(device))

    # Image downloading
    downloader = ImageDownloader(location=location,
                                 zoom=zoom,
                                 secret_key=secret_key)

    # Query
    query = ModelQuery(model=model,
                       quantile=quantile,
                       device=device,
                       transform=transform
    )

    # Downlad the image and query the model.
    x = downloader.request()
    pred, score, cam = query.make_query(x.transpose(2,0,1))

    # Create output image in case of positive classification.
    final_image = None
    if cam is not None:
        final_image = overlay_image_mask(x, cam, 0.5, 'jet')

    # Response generation
    response = {'input': {'image': x,
                          'q': quantile,
                          'zoom': zoom,
                          'location': location
                         },

                'output': {'prediction':pred,
                           'score': round(score, 2),
                           'image':final_image
                          }
               }

    # Reading the response
    print('Pred: ', response['output']['prediction'])
    print('Score: ', response['output']['score'])
    print('Input q: ', response['input']['q'])
    print('Input zoom: ', response['input']['zoom'])
    print('Input loc: ', response['input']['location'])

    if response['output']['image'] is not None:
        print('Final image with detection', response['output']['image'].shape)
        plt.imshow(response['output']['image'])
        plt.axis('off')
        plt.savefig('output.png')
    else:
        print('No detection', response['input']['image'].shape)
        plt.imshow(response['input']['image'])
        plt.axis('off')
        plt.savefig('output.png')

#        plt.imshow(response['input'])
#        plt.axis('off')
#        plt.show() 
