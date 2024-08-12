import argparse
import logging
import os
import re

import cv2
from numpy import ndarray, dtype, floating, float_
from numpy._typing import _64Bit
from rasterio import DatasetReader
from rasterio.io import DatasetWriter
from tqdm import tqdm

from apriltags_utils import *
from orthophoto_utils import *
from utils.iterators import get_next
from utils.polygons import shrink_or_swell_shapely_polygon

logger = logging.getLogger(__name__)


def extract_intensities_from_orthophotos(batch_of_orthophotos: list[Path],
                                         paths_with_visibility: dict[Path, ndarray],
                                         panel_locations: list[GeoDataFrame],
                                         number_of_bands: int) -> ndarray[Any, dtype[floating[_64Bit] | float_]]:
    """
    This function extracts intensities from the orthophotos.
    It does so by looking at each photo and determining which panels are visible in the photo.
    If a panel is not visible in the image it is not visible in any of the bands
    as the photos are assumed to rectified and aligned.
    Therefore, extraction is skipped for that panel with `np.Nan` values saved in the output.
    Otherwise, the function uses the recorded gps location of the panel given by `panel_locations`
    to extract the mean intensity of the panel in that band for that photo
    :type number_of_bands: int
    :return: The extracted intensities for all orthophotos, with np.Nan for values that could not be found.
    """
    intensities = np.zeros((len(batch_of_orthophotos), len(panel_locations), number_of_bands))
    for photo_index, orthophoto_path in enumerate(tqdm(batch_of_orthophotos)):
        for panel_index, panel_location in enumerate(panel_locations):
            if not paths_with_visibility[orthophoto_path][panel_index]:
                intensities[photo_index][panel_index] = np.full(number_of_bands, np.NaN)
                continue
            # extract the mean intensity for each band at that panel location
            orthophoto: DatasetReader
            with rasterio.open(orthophoto_path) as orthophoto:
                panel_intensities_per_band = extract_using_geolocation(orthophoto, panel_location)
            intensities[photo_index][panel_index] = panel_intensities_per_band
    return intensities


def interpolate_intensities(intensities: ndarray[Any, dtype[floating[_64Bit] | float_]],
                            number_of_bands: int) -> ndarray[
    Any, dtype[floating[_64Bit] | float_]]:
    """
    This function is used to piecewise linearly interpolate the intensity values to fill the `np.Nan` gaps in the data.
    To interpolate we select all the values captured in all the images for a given panel and band.
    Only for photos where the panel was visible we have a value for the given band.
    8Bit Data might look like this: [np.NaN, np.NaN, 240.0, 241.0, 240.0, np.NaN, 242.0, np.NaN, np.NaN]
    After interpolation:            [240.00, 240.00, 240.0, 241.0, 240.0, 241.00, 242.0, 242.00, 242.00]
    :param number_of_bands: number of bands in the images
    :param panel_locations: location of the edges of the panels
    :rtype: ndarray[Any, dtype[floating[_64Bit] | float_]]
    :param intensities: intensity values matrix of shape (photo, panel, band) with some values being np.NaN.
    :return: The interpolated intensity values
    """
    for panel_index, _ in enumerate(panel_properties):
        for band_index in range(0, number_of_bands):
            intensities[:, panel_index, band_index] = interpolate(intensities[:, panel_index, band_index])
    return intensities


def convert_orthophotos_to_reflectance(paths: list[Path],
                                       intensities: ndarray[Any, dtype[floating[_64Bit] | float_]]) -> \
        list[list[ndarray]]:
    """
    This function converts the intensity values to reflectance values.
    For each photo we convert each band separately
    by collecting all the intensities of the panels for the given photo and band.
    The intensities are then combined with the known reflectance values of the panels
    at the given band to fit a linear function (Empirical Line Method).
    Read more about ELM: https://www.asprs.org/wp-content/uploads/2015/05/3E%5B5%5D-paper.pdf
    :param panel_properties: list of intensities for each band for each panel
    :param paths: List of orthophoto paths
    :param intensities: intensity values matrix of shape (photo, panel, band)
    :return: List of reflectance photos, each photo is a list of bands, each band is a ndarray of shape (width, height)
    """
    unconverted_photos = []
    converted_photos = []
    for photo_index, orthophoto_path in enumerate(tqdm(paths)):
        converted_bands = []
        orthophoto: DatasetReader
        with rasterio.open(orthophoto_path) as orthophoto:
            photo: ndarray = orthophoto.read()
        for band_index, band in enumerate(photo):
            intensities_of_panels = intensities[photo_index, :, band_index]
            if np.isnan(intensities_of_panels).any():
                # If for some reason not all intensities are present, we save the indices for debugging purposes
                # A None value is appended to the converted photos to not lose
                # the connection between orthophotos and converted photos based on the index
                unconverted_photos.append((photo_index, band_index))
                converted_photos.append(None)
                break
            coefficients = fit(intensities_of_panels, get_band_reflectance(panel_properties, band_index))
            converted_bands.append(convert(band, coefficients))
        else:
            # For loop did not break, therefore all bands were converted
            converted_photos.append(converted_bands)
    if len(unconverted_photos) > 0:
        print("WARNING: Could not convert", len(unconverted_photos), "photos")
    return converted_photos


def convert_images_to_reflectance(paths: list[Path],
                                  intensities: ndarray[Any, dtype[floating[_64Bit] | float_]],
                                  band_index: int) -> list[ndarray | None]:
    """
    This function converts the intensity values to reflectance values.
    For each photo we convert each band separately
    by collecting all the intensities of the panels for the given photo and band.
    The intensities are then combined with the known reflectance values of the panels
    at the given band to fit a linear function (Empirical Line Method).
    Read more about ELM: https://www.asprs.org/wp-content/uploads/2015/05/3E%5B5%5D-paper.pdf
    :param band_index: index of the band of the image
    :param panel_properties: list of intensities for each band for each panel
    :param paths: List of image paths
    :param intensities: intensity values matrix of shape (photo, panel, band)
    :return: List of reflectance photos, each photo is a list of bands, each band is a ndarray of shape (width, height)
    """
    unconverted_photos = []
    converted_photos = []
    for image_index, path in enumerate(tqdm(paths)):
        intensities_of_panels = intensities[image_index, :]
        if np.isnan(intensities_of_panels).any():
            # If for some reason not all intensities are present, we save the indices for debugging purposes
            # A None value is appended to the converted photos to not lose
            # the connection between orthophotos and converted photos based on the index
            unconverted_photos.append(image_index)
            converted_photos.append(None)
            break
        coefficients = fit(intensities_of_panels, get_band_reflectance(panel_properties, band_index))
        band = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        converted_photos.append(convert(band, coefficients))

    if len(unconverted_photos) > 0:
        print("WARNING: Could not convert", len(unconverted_photos), "photos")
    return converted_photos


def save_orthophotos(paths: list[Path], converted_photos: list[list[ndarray]]) -> None:
    """
    This function saves the converted photos as .tif files into a new "/transformed/" directory in the images folder
    :param paths: List of orthophoto paths
    :param converted_photos: List of reflectance photos, each photo is a list of bands,
    each band is a ndarray of shape (width, height)
    """
    for path, photo in zip(paths, converted_photos):
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
        save_bands(output_path, photo, meta)


def save_images(paths: list[Path], converted_photos: list[ndarray]) -> None:
    """
    This function saves the converted photos as .tif files into a new "/transformed/" directory in the images folder
    :param paths: List of orthophoto paths
    :param converted_photos: List of reflectance photos, each photo is a list of bands,
    each band is a ndarray of shape (width, height)
    """
    for path, photo in zip(paths, converted_photos):
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
        dst: DatasetWriter
        with rasterio.open(output_path, 'w', **meta) as dst:
            dst.write_band(1, photo)


def build_batches_per_full_visibility(paths_with_visibility: dict[Path, ndarray]) -> list[list[Path]]:
    batches = []
    current_batch = []
    for (path, panel_visibility), (next_path, next_panel_visibility) in tqdm(get_next(paths_with_visibility.values())):
        current_batch.append(path)

        all_panels_visible = panel_visibility.all()
        if all_panels_visible:
            batches.append(current_batch)

            if next_panel_visibility.all():
                current_batch = []
            else:
                # This results in photos at the end of a batch being also in the next batch.
                # Therefore they are calculated twice, but otherwise the interpolation would lose their information.
                # The next batch could start with a images without panels in it,
                # which would need the image with panels before it to be interpolated correctly
                current_batch = [path]
    batches.append(current_batch)
    return batches


def build_batches_per_band(paths: list[Path]) -> dict[str, list[Path]]:
    # TODO add visibility batching also
    batches = dict()
    # Regular expression to match file names and capture the base and suffix
    pattern = re.compile(r".*_(\d+)\.tif$")

    for image_path in paths:
        match = pattern.match(image_path.name)
        if match:
            band_index = match.group(1)
            batches[band_index].append(image_path)

    return batches


def extract_using_apriltags(path):
    detector_config = AprilTagDetector.Config()
    detector_config.quadDecimate = 1.0
    detector_config.numThreads = 4
    detector_config.refineEdges = 1.0

    # TODO: add to args
    horizontal_focal_length_pixels = 1581.7867974691412
    horizontal_focal_center_pixels = 678.6724626822399
    vertical_focal_length_pixels = 1581.7867974691412
    vertical_focal_center_pixels = 529.4318832108801
    tag_size = 0.2
    sensor_size_mm = 6.3
    focal_length_mm = 5.5
    panel_size_m = 0.8

    estimator_config = AprilTagPoseEstimator.Config(tag_size, horizontal_focal_length_pixels,
                                                    vertical_focal_length_pixels,
                                                    horizontal_focal_center_pixels, vertical_focal_center_pixels)

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    panel_intensities = []

    for panel in panel_properties:
        d = AprilTagDetector()
        d.addFamily(panel.family)
        d.setConfig(detector_config)
        if args.detection_type not in ['apriltag_corner_tags', 'apriltag_single_tag']:
            raise ValueError("Unknown Detection Type for Apriltag algorithm")

        if args.detection_type == 'apriltag_corner_tags':
            if not isinstance(panel.tags, list) or len(panel.tags) != 4:
                raise ValueError(
                    "panel properties need to specify panel corners in apriltag_edges mode. Change your panel properties file")
            (tags, corners) = get_panels_ct(img, d, detector_config, )
            assert len(tags) == len(panel.tags)

        if args.detection_type == 'apriltag_single_tag':
            if isinstance(panel.tags, list):
                raise ValueError(
                    "Only one tag per panel in apriltag_single mode. Change your panel properties file")
            (tag, corners) = get_panels_st(img, d, sensor_size_mm, focal_length_mm, panel_size_m, estimator_config,
                                           list(panel.single_tag))

        polygon = shapely.Polygon(corners)
        polygon = shrink_or_swell_shapely_polygon(polygon, 0.2)
        mask = rasterize([polygon], out_shape=img.shape)
        # mean the radiance values to get a radiance value for the detection
        mean = np.ma.array(img, mask=~(mask.astype(np.bool_))).mean()
        panel_intensities.append(mean)
    return panel_intensities


def extract_intensities_from_apriltags(batch):
    intensities = np.zeros((len(batch), len(panel_properties)))
    for img_index, path in enumerate(tqdm(batch)):
        panel_intensities = extract_using_apriltags(path)
        intensities[img_index] = panel_intensities
    return intensities


def orthophoto_main():
    # python src/main.py "data/20240529_uav_multispectral_orthos_20m/orthophotos" "reflectance_panel_example_data.json" "data/20240529_uav_multispectral_orthos_20m/20240529_tarps_locations.gpkg"

    panel_locations = load_panel_locations(Path(args.panel_locations))
    orthophoto_paths = get_orthophoto_paths(args.images)

    ophoto: DatasetReader
    with rasterio.open(orthophoto_paths[0]) as ophoto:
        number_of_bands = len(ophoto.read())

    # --- Input validation
    assert len(panel_properties) == len(panel_locations)
    assert number_of_bands == len(panel_properties[0]['bands'])

    paths_with_visibility = {}
    for path in tqdm(orthophoto_paths):
        panels_visible = np.array(
            [is_panel_in_orthophoto(path, panel) for panel in panel_locations]
        )
        paths_with_visibility[path] = panels_visible

    batches = build_batches_per_full_visibility(paths_with_visibility)

    for batch in batches:
        # --- Run pipeline
        i = extract_intensities_from_orthophotos(batch, paths_with_visibility, panel_locations, number_of_bands)
        i = interpolate_intensities(i, number_of_bands)
        c = convert_orthophotos_to_reflectance(batch, i)
        del i
        save_orthophotos(batch, c)
        del c
        del batch


def get_apriltag_paths(dir: Path) -> List[Path]:
    template_path = Path(dir).resolve()
    return list(sorted([filepath for filepath in template_path.glob("*.tif")]))


def apriltag_main():
    img_paths = get_apriltag_paths(args.images)
    batches = build_batches_per_band(img_paths)
    number_of_bands = len(batches.keys())

    # --- Input validation
    assert number_of_bands == len(panel_properties[0]['bands'])

    for (band_name, batch) in batches.values():
        # --- Run pipeline
        i = extract_intensities_from_apriltags(batch)
        for panel_index, _ in enumerate(panel_properties):
            i[:, panel_index] = interpolate(i[:, panel_index])
        c = convert_images_to_reflectance(batch, i, int(band_name))  # TODO check bandname conversion
        del i
        save_images(batch, c)
        del c
        del batch


if __name__ == '__main__':
    # --- Get input arguments from user
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        prog='ReflectDetect',
        description='Automatically detect reflection calibration panels in images and transform the given images to '
                    'reflectance',
        epilog='If you have any questions, please contact')
    parser.add_argument("images", help="Path to the image files", type=str)
    parser.add_argument("detection_type", help="Type of detection ('apriltag_edges', 'apriltag_single', 'geolocation')",
                        type=str)
    parser.add_argument("panel_properties", help="Path to the property file of the panels", type=str)
    parser.add_argument("panel_locations", help="Path to the GeoPackage file", type=str)
    parser.add_argument("family", help="Name of the apriltag family ('tag25h9', 'tagStandard41h12', ...)", type=str)
    args = parser.parse_args()

    panel_properties = load_panel_properties(args.panel_properties)

    detection_type: str = args.detection_type

    if detection_type not in ['apriltag_corner_tags', 'apriltag_single_tag', 'geolocation']:
        raise ValueError('detection_type', detection_type, 'unknown')

    if detection_type.startswith("apriltag_"):
        apriltag_main()
    if detection_type == 'geolocation':
        orthophoto_main()
