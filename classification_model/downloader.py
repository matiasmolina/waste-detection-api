# Matias D. Molina - molinamatiasd@gmail.com - linkedin.com/in/matiasmolina

import io
from typing import Union, Optional, Tuple
from PIL import Image
import numpy as np
import requests

class ImageDownloader():
    def __init__(self, location: Tuple[float, float],
                       zoom: int = 18,
                       size: int = 512,
                       maptype: str = 'satellite',
                       secret_key: str = ''):

        self.center = location
        self.zoom = zoom
        self.size = size
        self.maptype = maptype
        self.key = secret_key

    def _set_secret_key(self, value: str):
        self.key = value
    
    def image_bytes_to_numpy(self, image_bytes: bytes) -> np.ndarray:
        image = Image.open(io.BytesIO(image_bytes))
        image = image.convert('RGB')
        image = np.array(image)
        return image

    def generate_url(self) -> str:          
        center = ','.join(tuple(map(str, self.center)))

        url = 'https://maps.googleapis.com/maps/api/staticmap?'
        url += f'center={center}'
        url += f'&zoom={self.zoom}'
        url += f'&size={self.size}x{self.size}'
        url += f'&maptype={self.maptype}'
        url += f'&key={self.key}'

        self.url = url
        return url

    def request(self, return_as_numpy=True):
        url = self.generate_url()        
        try:
            response = requests.get(url)
            if response.status_code == 200:
                if return_as_numpy:
                    return self.image_bytes_to_numpy(response.content)
                else:
                    return response.content
            else:
                raise Exception('Error when getting the image. Code: ', 
                                 response.status_code)
        except requests.exceptions.HTTPError as err:
            raise SystemExit(err)
