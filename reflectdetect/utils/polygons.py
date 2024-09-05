import numpy as np
from shapely import Polygon, centroid


def shrink_shapely_polygon(polygon: Polygon, factor: float = 0.80) -> Polygon:
    """returns the shapely polygon which is smaller passed by factor.

    :param polygon: a shapely polygon to shrink
    :param factor: the factor to shrink the polygon by. Default value is 0.80 resulting in a polygon with 80% the size
    :return: the shrunken polygon
    """
    center = np.squeeze(np.array(centroid(polygon).coords.xy))

    return Polygon((np.array(polygon.exterior.coords.xy).T - center) * factor + center)
