import math

import numpy as np


def convert_resolution_unit(resolution, unit):
    """
    Convert focal plane resolution to pixels per millimeter.

    Parameters:
        resolution (float): Focal plane resolution.
        unit (int): The unit of resolution (EXIF:FocalPlaneResolutionUnit).

    Returns:
        float: Focal plane resolution in pixels per millimeter.
    """
    if unit == 1:  # No unit, not common, assume pixels per millimeter
        return resolution
    elif unit == 2:  # Inches to millimeters
        return resolution / 25.4
    elif unit == 3:  # Centimeters to millimeters
        return resolution / 10.0
    elif unit == 4:  # Already in millimeters
        return resolution
    elif unit == 5:  # Micrometers to millimeters
        return resolution * 1000.0
    else:
        raise ValueError("Unknown FocalPlaneResolutionUnit")


def calculate_sensor_size(image_resolution, focal_plane_x_res, focal_plane_y_res, focal_plane_resolution_unit):
    """
    Calculate the sensor size in millimeters using focal plane resolutions.

    Parameters:
        image_resolution (tuple): Image resolution (width, height) in pixels.
        focal_plane_x_res (float): Focal plane X resolution.
        focal_plane_y_res (float): Focal plane Y resolution.
        focal_plane_resolution_unit (int): The unit of the focal plane resolution.

    Returns:
        tuple: Sensor width and height in millimeters.
    """
    # Convert resolution to pixels per millimeter
    focal_plane_x_res_mm = convert_resolution_unit(focal_plane_x_res, focal_plane_resolution_unit)
    focal_plane_y_res_mm = convert_resolution_unit(focal_plane_y_res, focal_plane_resolution_unit)

    # Calculate sensor dimensions
    sensor_width_mm = image_resolution[0] / focal_plane_x_res_mm
    sensor_height_mm = image_resolution[1] / focal_plane_y_res_mm
    return sensor_width_mm, sensor_height_mm


def calculate_panel_size_in_pixels(altitude, resolution, physical_panel_size, focal_length_mm, focal_plane_x_res,
                                   focal_plane_y_res, focal_plane_resolution_unit):
    """
    Calculate the expected size of an object in pixels based on camera parameters and object physical size.

    Parameters:
        altitude (float): Altitude in meters.
        resolution (tuple): Image resolution (width, height) in pixels.
        focal_length_mm (float): Focal length in millimeters.
        physical_panel_size (tuple): Physical size of the object in meters (width, height).
        focal_plane_x_res (float): Focal plane X resolution.
        focal_plane_y_res (float): Focal plane Y resolution.
        focal_plane_resolution_unit (int): The unit of the focal plane resolution.

    Returns:
        tuple: Expected width and height of the object in pixels.
    """
    resolution = (float(resolution[0]), float(resolution[1]))

    # Calculate the sensor size
    sensor_width_mm, sensor_height_mm = calculate_sensor_size(resolution, focal_plane_x_res, focal_plane_y_res,
                                                              focal_plane_resolution_unit)
    # Convert sensor dimensions to meters
    sensor_width_m = sensor_width_mm / 1000  # Convert mm to m
    sensor_height_m = sensor_height_mm / 1000
    focal_length_m = (focal_length_mm / 1000)

    # Calculate horizontal and vertical Field of View (FoV)
    fov_horizontal = 2 * math.atan(sensor_width_m / (2 * focal_length_m))
    fov_vertical = 2 * math.atan(sensor_height_m / (2 * focal_length_m))

    # Calculate scale in pixels per meter for width and height
    scale_pixels_per_meter_width = resolution[0] / (altitude * math.tan(fov_horizontal / 2))
    scale_pixels_per_meter_height = resolution[1] / (altitude * math.tan(fov_vertical / 2))

    # Calculate expected panel size in pixels
    panel_width_pixels = np.intp(physical_panel_size[0] * scale_pixels_per_meter_width)
    panel_height_pixels = np.intp(physical_panel_size[1] * scale_pixels_per_meter_height)

    scale = 0.8
    return panel_width_pixels * scale, panel_height_pixels * scale


def get_panel_factors_for_band(panel_data, band):
    return [panel["bands"][band]["factor"] for panel in panel_data]
