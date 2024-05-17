import numpy as np
from rasterio.features import rasterize
import shapely as sg


# Expects panel locations to be a list of YOLO_OBB bounding boxes
# in the form [id, class, x1, y1, x2, y1, x3, y3, x4, y4]
def get_mean_radiance_values(panel_locations, img):
    panel_radiance_values = []
    for detection in panel_locations:
        # ignore id and class
        detection = detection[2:]
        # convert [x1, y1, x2, y2] to [(x1, y1), (x2, y2)] and instantiate polygon
        polygon = sg.Polygon(list(zip(detection, detection[1:]))[::2])
        mask = rasterize([polygon], out_shape=img.shape)
        # mean the radiance values to get a radiance value for the detection
        mean = np.ma.array(img, mask=~(mask.astype(np.bool_))).mean()
        panel_radiance_values.append(mean)
    return panel_radiance_values
