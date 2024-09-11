# Matias D. Molina - molinamatiasd@gmail.com - linkedin.com/in/matiasmolina

# Secret Key for GoogleMaps API.
secret_key = ''

# Trained model path.
model_path = './classification_model/models/model_cls.pt'

# Default resolution (zoom) for image downaloading.
# Valid numbers are 17,18,19.
# It could be extended to a wider range in case of changing the model.
default_zoom = 18

# Default quantile value for pixel-wise explainability.
# it could be a number in {0.90, 0.92, 0.93, 0.94, 0.96, 0.98}
default_pixel_quantile = 0.94

# Custom messages returned after classification.
positive_message = 'Positive detection.'
negative_message = 'Negative detection.'

# Server settings (also in set it in Dockerfile)
port_number = 5000

