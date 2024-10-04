import numpy as np
from numpy.typing import NDArray

from reflectdetect.PanelProperties import (
    ValidatedApriltagPanelProperties,
    ValidatedGeolocationPanelProperties,
)


def convert_resolution_unit(resolution: float, unit: int) -> float:
    """
    Convert focal plane resolution to pixels per millimeter.

    :param resolution: Focal plane resolution.
    :param unit:  The unit of resolution (EXIF:FocalPlaneResolutionUnit).
    :return: Focal plane resolution in pixels per millimeter.

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


def calculate_sensor_size(
        image_resolution: tuple[int, int],
        focal_plane_x_res: float,
        focal_plane_y_res: float,
        focal_plane_resolution_unit: int,
) -> tuple[float, float]:
    """
    Calculate the sensor size in millimeters using focal plane resolutions.

    :param image_resolution: Image resolution (width, height) in pixels.
    :param focal_plane_x_res: Focal plane X resolution.
    :param focal_plane_y_res: Focal plane Y resolution.
    :param focal_plane_resolution_unit: The unit of the focal plane resolution.
    :return: Sensor width and height in millimeters.
    """
    # Convert resolution to pixels per millimeter
    focal_plane_x_res_mm = convert_resolution_unit(
        focal_plane_x_res, focal_plane_resolution_unit
    )
    focal_plane_y_res_mm = convert_resolution_unit(
        focal_plane_y_res, focal_plane_resolution_unit
    )

    # Calculate sensor dimensions
    sensor_width_mm = image_resolution[0] / focal_plane_x_res_mm
    sensor_height_mm = image_resolution[1] / focal_plane_y_res_mm
    return sensor_width_mm, sensor_height_mm


def calculate_panel_size_in_pixels(
        altitude: float,
        resolution: tuple[int, int],
        physical_panel_size: tuple[float, float],
        focal_length_mm: float,
        focal_plane_x_res: float,
        focal_plane_y_res: float,
        focal_plane_resolution_unit: int,
        smudge_factor: float = 0.8,
) -> tuple[int, int]:
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
        smudge_factor (float, optional): Adjustment factor to correct for systematic error caused by difference between
        reported camera hyperparameters and actual values. Default is 0.8.

    Returns:
        tuple: Expected width and height of the object in pixels.
    """

    # Calculate the sensor size
    sensor_width_mm, sensor_height_mm = calculate_sensor_size(
        resolution, focal_plane_x_res, focal_plane_y_res, focal_plane_resolution_unit
    )
    # Convert sensor dimensions to meters
    sensor_width_m = sensor_width_mm / 1000
    sensor_height_m = sensor_height_mm / 1000
    focal_length_m = focal_length_mm / 1000

    # Calculate Ground Sample Distance (GSD)
    gsd_width = (altitude * sensor_width_m) / focal_length_m
    gsd_height = (altitude * sensor_height_m) / focal_length_m

    # Calculate scale in pixels per meter for width and height
    scale_pixels_per_meter_width = float(resolution[0]) / gsd_width
    scale_pixels_per_meter_height = float(resolution[1]) / gsd_height

    # Apply smudge factor to correct for the observed discrepancy
    scale_pixels_per_meter_width *= smudge_factor
    scale_pixels_per_meter_height *= smudge_factor

    # Calculate expected panel size in pixels
    panel_width_pixels = int(physical_panel_size[0] * scale_pixels_per_meter_width)
    panel_height_pixels = int(physical_panel_size[1] * scale_pixels_per_meter_height)

    return panel_width_pixels, panel_height_pixels


def get_panel_intensity(intensity_values: np.ma.MaskedArray) -> float:
    if intensity_values.mask.all():
        # The whole array is masked
        return np.nan
    intensity_values = intensity_values.astype(np.float64)
    intensity_values = np.ma.filled(intensity_values, np.nan)
    q95, q5 = np.nanpercentile(intensity_values, [95, 5])
    lower_bound = q5
    upper_bound = q95
    # remove outliers and mean
    intensity_values = intensity_values[intensity_values >= lower_bound]
    intensity_values = intensity_values[intensity_values <= upper_bound]

    return float(np.nanmean(intensity_values))


def get_band_reflectance(
        panels_properties: list[ValidatedGeolocationPanelProperties] | list[ValidatedApriltagPanelProperties],
        band_index: int,
) -> NDArray[np.float64]:
    """

    :param panels_properties: band reflectance values
    :param band_index: which band to use
    :return: the reflectance values of each panel at a given band
    """
    return np.array([properties.bands[band_index] for properties in panels_properties])
