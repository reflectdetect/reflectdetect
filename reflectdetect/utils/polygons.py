from shapely import geometry, Polygon


# https://stackoverflow.com/questions/49558464/shrink-polygon-using-corner-coordinates
def shrink_or_swell_shapely_polygon(polygon: Polygon, factor: float = 0.10, swell: bool = False) -> Polygon:
    ''' returns the shapely polygon which is smaller or bigger by passed factor.
        If swell = True , then it returns bigger polygon, else smaller '''

    xs = list(polygon.exterior.coords.xy[0])
    ys = list(polygon.exterior.coords.xy[1])
    x_center = 0.5 * min(xs) + 0.5 * max(xs)
    y_center = 0.5 * min(ys) + 0.5 * max(ys)
    min_corner = geometry.Point(min(xs), min(ys))
    center = geometry.Point(x_center, y_center)
    shrink_distance = center.distance(min_corner) * factor

    if swell:
        return polygon.buffer(shrink_distance)  # expand
    else:
        return polygon.buffer(-shrink_distance)  # shrink
