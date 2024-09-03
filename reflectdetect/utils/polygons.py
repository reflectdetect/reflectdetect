import numpy as np
from shapely import Polygon, centroid


# https://stackoverflow.com/questions/49558464/shrink-polygon-using-corner-coordinates
def shrink_shapely_polygon(polygon: Polygon, factor: float = 0.80) -> Polygon:
    ''' returns the shapely polygon which is smaller passed by factor. '''
    center = np.squeeze(np.array(centroid(polygon).coords.xy))

    return Polygon((np.array(polygon.exterior.coords.xy).T - center) * factor + center)
