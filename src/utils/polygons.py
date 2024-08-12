# https://stackoverflow.com/questions/49558464/shrink-polygon-using-corner-coordinates
def shrink_or_swell_shapely_polygon(my_polygon, factor=0.10, swell=False):
    ''' returns the shapely polygon which is smaller or bigger by passed factor.
        If swell = True , then it returns bigger polygon, else smaller '''
    from shapely import geometry

    # my_polygon = mask2poly['geometry'][120]

    shrink_factor = 0.10  # Shrink by 10%
    xs = list(my_polygon.exterior.coords.xy[0])
    ys = list(my_polygon.exterior.coords.xy[1])
    x_center = 0.5 * min(xs) + 0.5 * max(xs)
    y_center = 0.5 * min(ys) + 0.5 * max(ys)
    min_corner = geometry.Point(min(xs), min(ys))
    max_corner = geometry.Point(max(xs), max(ys))
    center = geometry.Point(x_center, y_center)
    shrink_distance = center.distance(min_corner) * 0.10

    if swell:
        my_polygon_resized = my_polygon.buffer(shrink_distance)  # expand
    else:
        my_polygon_resized = my_polygon.buffer(-shrink_distance)  # shrink

    # visualize for debugging
    # x, y = my_polygon.exterior.xy
    # plt.plot(x,y)
    # x, y = my_polygon_shrunken.exterior.xy
    # plt.plot(x,y)
    ## to net let the image be distorted along the axis
    # plt.axis('equal')
    # plt.show()

    return my_polygon_resized