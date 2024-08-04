import argparse
import logging
import os
from pathlib import Path

import numpy as np
import rasterio
from tqdm import tqdm

from orthophoto_utils import load_panel_properties, load_panel_locations, get_orthophoto_paths, is_panel_in_orthophoto, \
    extract, interpolate, fit, get_band_reflectance, convert, save

logger = logging.getLogger(__name__)


def extract_intensities():
    intensities = np.zeros((len(orthophoto_paths), len(panel_locations), number_of_bands))
    for photo_index, orthophoto_path in enumerate(tqdm(orthophoto_paths)):
        with rasterio.open(orthophoto_path) as orthophoto:
            photo = orthophoto.read()
            for panel_index, panel in enumerate(panel_locations):
                if not is_panel_in_orthophoto(orthophoto_path, panel):
                    intensities[photo_index][panel_index] = np.full(len(photo), np.NaN)
                    continue
                panel_intensities_per_band = extract(orthophoto, panel)
                intensities[photo_index][panel_index] = panel_intensities_per_band
    return intensities


def interpolate_intensities(intensities):
    for panel_index, panel in enumerate(panel_locations):
        for band_index in range(0, number_of_bands):
            intensities[:, panel_index, band_index] = interpolate(intensities[:, panel_index, band_index])
    return intensities


def convert_photos(intensities):
    unconvertable_photos = []
    converted_photos = []
    for photo_index, orthophoto_path in enumerate(tqdm(orthophoto_paths)):
        converted_bands = []
        with rasterio.open(orthophoto_path) as orthophoto:
            photo = orthophoto.read()
        for band_index, band in enumerate(photo):
            intensities_of_panels = intensities[photo_index, :, band_index]
            if np.isnan(intensities_of_panels).any():
                unconvertable_photos.append((photo_index, band_index))
                converted_photos.append(None)
                break
            coeffs = fit(intensities_of_panels, get_band_reflectance(panel_properties, band_index))
            converted_bands.append(convert(band, coeffs))
        else:
            # for loop did not break, all bands were converted
            converted_photos.append(converted_bands)
    return converted_photos


def save_images(converted_photos):
    for path, photo in zip(orthophoto_paths, converted_photos):
        if photo is None:
            continue
        filename = path.as_posix().split("/")[-1].split(".")[0] + "_reflectance.tif"
        output_folder = "/".join(path.as_posix().split("/")[:-1]) + "/transformed/"
        os.makedirs(output_folder, exist_ok=True)
        output_path = output_folder + filename
        with rasterio.open(path) as original:
            meta = original.meta
        meta.update(
            dtype=rasterio.float32,
        )
        save(output_path, photo, meta)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        prog='ReflectDetect',
        description='Automatically detect reflection calibration panels in images and transform the given images to '
                    'reflectance',
        epilog='If you have any questions, please contact')
    parser.add_argument("images", help="Path to the image files", type=str)
    parser.add_argument("panel_properties", help="Path to the property file of the panels", type=str)
    parser.add_argument("panel_locations", help="Path to the GeoPackage file", type=str)
    args = parser.parse_args()

    # python src/main.py "data/20240529_uav_multispectral_orthos_20m/orthophotos" "reflectance_panel_example_data.json" "data/20240529_uav_multispectral_orthos_20m/20240529_tarps_locations.gpkg"

    # panel_properties_file = "../reflectance_panel_example_data.json"
    # geopackage_file = "../data/20240529_uav_multispectral_orthos_20m/20240529_tarps_locations.gpkg"
    # orthophotos_dir = "../data/20240529_uav_multispectral_orthos_20m/orthophotos"

    panel_properties = load_panel_properties(args.panel_properties)
    panel_locations = load_panel_locations(Path(args.panel_locations))
    orthophoto_paths = list(sorted(get_orthophoto_paths(args.images)))

    with rasterio.open(orthophoto_paths[0]) as orthophoto:
        number_of_bands = len(orthophoto.read())

    assert len(panel_properties) == len(panel_locations)
    assert number_of_bands == len(panel_properties[0]['bands'])

    i = extract_intensities()
    i = interpolate_intensities(i)
    c = convert_photos(i)
    save_images(c)
