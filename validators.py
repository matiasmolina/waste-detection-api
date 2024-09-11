# Matias D. Molina - molinamatiasd@gmail.com - linkedin.com/in/matiasmolina

def is_valid_latitude(lat):
    type_ok = type(lat) == float
    return type_ok and (-90 <= lat <= 90)

def is_valid_longitude(lon):
    type_ok = type(lon) == float
    return type_ok and (-180 <= lon <= 180)

def is_valid_zoom(zoom):
    return zoom in [17, 18, 19]

def is_valid_quantile(q):
    return q in [0.90, 0.92, 0.94, 0.96, 0.98]

def check_arguments(zoom, quantile, lat, lon):
    err_msg = {}
    if not is_valid_zoom(zoom):
        err_msg['zoom'] = f'Invalid zoom value ({zoom}).'
    if not is_valid_quantile(quantile):
        err_msg['quantile'] = f'Invalid quantile value ({quantile}).'
    if not is_valid_latitude(lat):
       err_msg['latitutde'] = f'Invalid latitude value ({lat}).'
    if not is_valid_longitude(lon):
        err_msg['longitude'] = f'Invalid longitude value ({lon}).'
    return err_msg
