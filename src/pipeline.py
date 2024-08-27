from typing import Any, List, Tuple

import numpy as np
from numpy import ndarray


def fit(intensities: ndarray[Any, np.dtype[np.float64]], expected_reflectances: List[float]) -> \
        Tuple[float, float]:
    slope, intersect = np.polyfit(intensities, expected_reflectances, 1)
    return slope, intersect


def interpolate(values: ndarray) -> ndarray:
    is_none = [np.isnan(v) for v in values]
    non_none_vals = [(i, v) for i, v in enumerate(values) if not np.isnan(v)]

    if len(non_none_vals) < 1:
        print('No values found for interpolation.')
        return values

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

                interpolated_values = [slope * i + intercept for i in range(lower_idx, upper_idx + 1)]

                # Update the values array with the interpolated values
                for j in range(lower_idx, upper_idx + 1):
                    values[j] = interpolated_values[j - lower_idx]
    return values


def convert(band_image: ndarray, coeffs: Tuple[float, float]) -> ndarray:
    # converts a photo based on a linear transformation.
    return np.poly1d(coeffs)(band_image)


def interpolate_intensities(intensities: ndarray[Any, np.dtype[np.float64]],
                            number_of_bands: int, panel_properties) -> ndarray[Any, np.dtype[np.float64]]:
    """
    This function is used to piecewise linearly interpolate the intensity values to fill the `np.Nan` gaps in the data.
    To interpolate we select all the values captured in all the images for a given panel and band.
    Only for photos where the panel was visible we have a value for the given band.
    8Bit Data might look like this: [np.NaN, np.NaN, 240.0, 241.0, 240.0, np.NaN, 242.0, np.NaN, np.NaN]
    After interpolation:            [240.00, 240.00, 240.0, 241.0, 240.0, 241.00, 242.0, 242.00, 242.00]
    :param number_of_bands: number of bands in the images
    :rtype: ndarray[Any, dtype[np.float64]]
    :param intensities: intensity values matrix of shape (photo, panel, band) with some values being np.NaN.
    :return: The interpolated intensity values
    """
    for panel_index, _ in enumerate(panel_properties):
        for band_index in range(0, number_of_bands):
            intensities[:, panel_index, band_index] = interpolate(intensities[:, panel_index, band_index])
    return intensities
