# Matias D. Molina - molinamatiasd@gmail.com - linkedin.com/in/matiasmolina

import torch                                                                  
import torch.nn as nn
import torchvision.transforms as transforms                                   
from torch.nn import functional as F

import numpy as np

from typing import Union, Optional, Tuple

class ModelQuery():
    def __init__(self, threshold: float = 0.5,
                       quantile: float = 0.94,
                       model: Union[nn.Module, None] = None,
                       transform: Union[nn.Sequential, None] = None,
                       device: str = 'cpu'):

        self.model = model.to(device)
        self.transform = transform
        self.threshold = threshold
        self.quantile = quantile
        self.device = device
        
        self._validate_quantile(self.quantile)
        self._validate_threshold(self.threshold)

    def set_quantile(self, value: float):
        ''' Set the pixel-wise scoring threshold.'''
        self._validate_quantile(value)
        self.quantile = value

    def _validate_quantile(self, value):
        if not isinstance(value, float):
            raise TypeError("Pixel-scoring threshold must be a float.")
        if not 0. <= value and value <= 1.:
            raise ValueError("Pixel-scoring threshold must be in range [0,1].")
        if value not in [0.90, 0.92, 0.94, 0.96, 0.98]:
            raise ValueError("Pixel-scoring threshold must be in {0.90, 0.92, 0.94, 0.96, 0.98}.")

    def _validate_threshold(self, value):
        if not isinstance(value, float):
            raise TypeError("The threshold must be a float.")
        if not 0. <= value and value <= 1.:
            raise ValueError("The threshold must be in range [0,1].")
    
    def compute_cam(self, image: np.array, name: str, fc_weights: np.array):
        fc_weights = self.model.fc.weight.unsqueeze(2).unsqueeze(3)
        cam = fc_weights * self._activation_cam[name].sum(1, keepdim=True)
        
        cam = torch.where(cam <= torch.quantile(cam, self.quantile), 0, cam)
        cam = F.interpolate(cam, image.shape[1:3], mode="bilinear", align_corners=True)
        cam = cam.detach().cpu().numpy()[0,0,:,:]
        
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
        return cam 

    def make_query(self, image: np.array):
        '''
        Query pipeline. 

        Parameters
        ----------
        image : ndarray
            3D float array (it represents an RGB Image).

        Returns
        -------
        prediction: bool
            True or False is the prediction is positive or negative respectively.
            It is defined by the predictive score.
        score : float
            Classification score.
        cam: ndarray
            In case of positive prediction, it returns the Explainability of the classification.
            Otherwise, None is returned.
        '''
 
        self.activation = {}
        def get_activation(name):
            def hook(model, input, output):        
                self.activation[name] = output.detach()
            return hook

        input_ = torch.tensor(np.expand_dims(image, axis=0)).float()
        if self.transform:
            input_ = self.transform(input_).to(self.device)

        self.model.eval()
        with torch.no_grad():
            self.model.layer4.register_forward_hook(get_activation("layer4"))
                        
            output = self.model(input_)
            score = torch.nn.functional.sigmoid(output)
            prediction = (score > self.threshold)

        if prediction:
            fc_weights = self.model.fc.weight.unsqueeze(2).unsqueeze(3)
            cam = (fc_weights * self.activation["layer4"]).sum(1, keepdim=True)

            cam = torch.where(cam <= torch.quantile(cam, self.quantile), 0, cam)
            cam = F.interpolate(cam, image.shape[1:3], mode="bilinear", align_corners=True)
            cam = cam.detach().cpu().numpy()[0,0,:,:]
            cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
            
            # Avoid saving hook in case of double-quering
            self.activation = {}
        else:
            cam = None

        return (prediction.squeeze().item(), score.squeeze().item(), cam)
