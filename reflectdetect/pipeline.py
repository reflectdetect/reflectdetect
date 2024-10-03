from typing import Any

import numpy as np
from numpy.typing import NDArray
from rich.progress import Progress

from reflectdetect.utils.debug import ProgressBar


def fit(x: NDArray[np.float64], y: NDArray[np.float64]) -> tuple[float, float]:
    """
    Fits a linear function to the data

    :param x: 1D float array
    :param y: 1D float array
    :return: slope and intercept of the linear function
    """
    slope, intersect = np.polyfit(x, y, 1)
    return slope, intersect


def interpolate(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Interpolates the values to remove all np.nan values.
    NaN values that only have a datapoint on one side take the value of the closest datapoint
    NaN values that have a datapoint on each side get linearly interpolated between the two datapoints

    :param values: 1D float array
    :return: 1D float array without np.Nan values
    """
    is_none = [np.isnan(v) for v in values]
    non_none_vals = [(i, v) for i, v in enumerate(values) if not np.isnan(v)]

    if len(non_none_vals) == 0:
        raise Exception("No values found for interpolation.")

    for i, _ in enumerate(values):
        if is_none[i]:  # If our value is None, interpolate
            # Find the closest indices with value on either side
            lower = list(idx for idx, v in non_none_vals if idx < i)
            upper = list(idx for idx, v in non_none_vals if idx > i)
            if len(lower) == 0 and len(upper) == 0:
                continue

            if len(lower) == 0:
                upper_idx = min(upper)
                upper_value = values[upper_idx]
                values[i] = upper_value
            elif len(upper) == 0:
                lower_idx = max(lower)
                lower_value = values[lower_idx]
                values[i] = lower_value
            else:
                lower_idx = max(lower)
                upper_idx = min(upper)

                lower_value = values[lower_idx]
                upper_value = values[upper_idx]

                slope = (upper_value - lower_value) / (upper_idx - lower_idx)
                intercept = lower_value - slope * lower_idx

                interpolated_values = [
                    slope * i + intercept for i in range(lower_idx, upper_idx + 1)
                ]

                # Update the values array with the interpolated values
                for j in range(lower_idx, upper_idx + 1):
                    values[j] = interpolated_values[j - lower_idx]
    return values


def convert(
        band_image: NDArray[Any], coefficients: tuple[float, float], mask: NDArray[np.uint8] | None = None
) -> NDArray[np.float64]:
    """
    Uses coefficients of a linear function to convert a 2D image

    :param band_image: 2D image of intensity values
    :param coefficients: slope and intercept of a linear function
    :return: the band image with every datapoint being transformed by the linear function
    """
    # converts a photo based on a linear transformation.
    if mask is not None:
        result = np.where(mask, np.poly1d(coefficients)(band_image), np.nan)
    else:
        result = np.poly1d(coefficients)(band_image)

    # Clip result values to be between 0 and 1 (reflectance)
    result[result < 0] = 0
    result[result > 1] = 1
    return result


def interpolate_intensities(
        intensities: NDArray[np.float64],
        number_of_bands: int,
        number_of_panels: int,
        progress: Progress | None,
) -> NDArray[np.float64]:
    """
    This function is used to piecewise linearly interpolate the intensity values to fill the `np.Nan` gaps in the data.
    To interpolate we select all the values captured in all the images for a given panel and band.
    Only for photos where the panel was visible we have a value for the given band.
    Data might look like this: [np.NaN, np.NaN, 240.0, 241.0, 240.0, np.NaN, 242.0, np.NaN, np.NaN]
    After interpolation:            [240.00, 240.00, 240.0, 241.0, 240.0, 241.00, 242.0, 242.00, 242.00]

    :param number_of_panels: number of panels
    :param number_of_bands: number of bands in the images
    :rtype: ndarray[Any, dtype[np.float64]]
    :param intensities: intensity values matrix of shape (photo, panel, band) with some values being np.NaN.
    :return: The interpolated intensity values
    """
    with ProgressBar(progress, "Interpolating", total=number_of_panels) as pb:
        for panel_index in range(0, number_of_panels):
            for band_index in range(0, number_of_bands):
                intensities[:, panel_index, band_index] = interpolate(
                    intensities[:, panel_index, band_index]
                )
            pb.update()
        return intensities
