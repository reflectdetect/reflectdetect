import argparse
import json
import logging
import re

import cv2
import shapely
from exiftool import ExifToolHelper
from numpy import ndarray, dtype
from rasterio.features import rasterize

from utils.apriltags import *
from utils.debug import debug_combine_and_plot_intensities, debug_save_intensities, debug_show_panel, \
    debug_save_intensities_single_band
from utils.iterators import get_next
from utils.orthophoto import *
from utils.panel import calculate_panel_size_in_pixels
from utils.paths import get_output_path
from utils.polygons import shrink_or_swell_shapely_polygon

logger = logging.getLogger(__name__)


def get_band_reflectance(panels_properties, band_index) -> list:
    # return the reflectance values of each panel at a given band
    return [properties['bands'][band_index] for properties in panels_properties]


def load_panel_properties(panel_properties_file):
    with open(panel_properties_file) as f:
        panels = json.load(f)
    return panels


def interpolate_intensities(intensities: ndarray[Any, dtype[np.float64]],
                            number_of_bands: int) -> ndarray[Any, dtype[np.float64]]:
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


def convert_orthophotos_to_reflectance(paths: list[Path],
                                       intensities: ndarray[Any, dtype[np.float64]]) -> \
        list[list[ndarray]]:
    """
    This function converts the intensity values to reflectance values.
    For each photo we convert each band separately
    by collecting all the intensities of the panels for the given photo and band.
    The intensities are then combined with the known reflectance values of the panels
    at the given band to fit a linear function (Empirical Line Method).
    Read more about ELM: https://www.asprs.org/wp-content/uploads/2015/05/3E%5B5%5D-paper.pdf
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
                                  intensities: ndarray[Any, dtype[np.float64]],
                                  band_index: int) -> list[ndarray | None]:
    """
    This function converts the intensity values to reflectance values.
    For each photo we convert each band separately
    by collecting all the intensities of the panels for the given photo and band.
    The intensities are then combined with the known reflectance values of the panels
    at the given band to fit a linear function (Empirical Line Method).
    Read more about ELM: https://www.asprs.org/wp-content/uploads/2015/05/3E%5B5%5D-paper.pdf
    :param band_index: index of the band of the image
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
        band = cv2.imread(path.as_posix(), cv2.IMREAD_GRAYSCALE)
        converted_photos.append(convert(band, coefficients))

    if len(unconverted_photos) > 0:
        print("WARNING: Could not convert", len(unconverted_photos), "photos")
    return converted_photos


def get_camera_properties(image_path):
    with ExifToolHelper() as et:
        metadata = et.get_metadata(image_path)[0]
        # focal_length_mm, focal_plane_x_res, focal_plane_y_res, focal_plane_resolution_unit
    return metadata["EXIF:FocalLength"], metadata["EXIF:FocalPlaneXResolution"], metadata["EXIF:FocalPlaneYResolution"], \
        metadata[
            "EXIF:FocalPlaneResolutionUnit"],


def extract_using_apriltags(path, detector, all_ids: list[int], panel_size_m: (float, float), tag_size_m):
    img = cv2.imread(path.as_posix(), cv2.IMREAD_GRAYSCALE)

    all_tags = detect_tags(img, detector, all_ids)
    if len(all_tags) != len(panel_properties):
        return [None] * len(panel_properties)

    altitude = get_altitude_from_panels(all_tags, path, (len(img[0]), len(img)), tag_size_m)

    resolution = (len(img[0]), len(img))
    properties = get_camera_properties(path)
    panel_size_pixel = calculate_panel_size_in_pixels(altitude, resolution, panel_size_m, *properties)
    panel_intensities = [None] * len(panel_properties)
    for tag in all_tags:
        panels = list(filter(lambda p: p["family"] == tag.getFamily() and p["single_tag"] == tag.getId(),
                             panel_properties))
        if not len(panels) == 1:
            raise Exception("Could not associate panel with found tag")
        panel_index = panel_properties.index(panels[0])
        corners = get_panel(tag, panel_size_pixel[0], resolution)

        if corners is None:
            continue
        else:
            if args.debug:
                output_path = get_output_path(path, "panel_" + str(tag.getId()) + "_" + tag.getFamily(),
                                              "debug/panels")
                debug_show_panel(img, [tag], corners, output_path)
            polygon = shapely.Polygon(corners)
            polygon = shrink_or_swell_shapely_polygon(polygon, 0.2)
            panel_mask = rasterize([polygon], out_shape=img.shape)
            mean = np.ma.array(img, mask=~(panel_mask.astype(np.bool_))).mean()
            panel_intensities[panel_index] = mean
    return panel_intensities


def extract_intensities_from_apriltags(batch, detector, all_ids, estimator_config, panel_size_m):
    intensities = np.zeros((len(batch), len(panel_properties)))
    for img_index, path in enumerate(tqdm(batch)):
        panel_intensities = extract_using_apriltags(path, detector, all_ids, estimator_config,
                                                    panel_size_m)
        intensities[img_index] = panel_intensities
    return intensities


def get_apriltag_paths(images_directory: Path) -> List[Path]:
    template_path = Path(images_directory).resolve()
    return list(sorted([filepath for filepath in template_path.glob("*.tif")]))


def build_batches_per_full_visibility(paths_with_visibility: dict[Path, ndarray]) -> list[list[Path]]:
    batches = []
    current_batch = []
    for (path, panel_visibility), nextVisibility in tqdm(get_next(paths_with_visibility.items())):
        current_batch.append(path)

        all_panels_visible = panel_visibility.all()
        if all_panels_visible:
            batches.append(current_batch)
            if nextVisibility is not None and nextVisibility[1].all():
                # The next image has all panels visible, so no need to include this one twice
                current_batch = []
            else:
                # This results in photos at the end of a batch being also in the next batch.
                # Therefore, they are calculated twice, but otherwise the interpolation would lose their information.
                # The next batch could start with an images without panels in it,
                # which would need the image with panels before it to be interpolated correctly
                current_batch = [path]
    batches.append(current_batch)
    return batches


def build_batches_per_band(paths: list[Path]) -> list[list[Path]]:
    # TODO also add visibility batching
    batches = []
    # Regular expression to match file names and capture the base and suffix
    pattern = re.compile(r".*_(\d+)\.tif$")

    for image_path in paths:
        match = pattern.match(image_path.name)
        if match:
            band_index = int(match.group(1)) - 1
            if band_index == len(batches):
                batches.append([])
            elif band_index > len(batches):
                raise ValueError(
                    "Problem with the sorting of the pats or regex")  # TODO better path parsing generalization
            batches[band_index].append(image_path)

    return batches


def orthophoto_main():
    panel_locations = load_panel_locations(Path(args.panel_locations))
    orthophoto_paths = get_orthophoto_paths(args.images)

    photo: DatasetReader
    with rasterio.open(orthophoto_paths[0]) as photo:
        number_of_bands = len(photo.read())

    # --- Input validation
    if len(panel_properties) != len(panel_locations):
        raise Exception("Number of panel specifications does not match number of panel locations")
    if number_of_bands != len(panel_properties[0]['bands']):
        raise Exception("Number of bands in the images does not match number of bands in the panel specification")

    paths_with_visibility = {}
    for path in tqdm(orthophoto_paths):
        panels_visible = np.array(
            [is_panel_in_orthophoto(path, p) for p in panel_locations]
        )
        paths_with_visibility[path] = panels_visible

    batches = build_batches_per_full_visibility(paths_with_visibility)

    output_folder = None
    if args.debug:
        output_folder = args.images + "/debug/intensity/"
        for p in Path(output_folder).glob("*.csv"):
            p.unlink()

    for batch in batches:
        # --- Run pipeline
        i = extract_intensities_from_orthophotos(batch, paths_with_visibility, panel_locations, number_of_bands)
        if args.debug:
            debug_save_intensities(i, number_of_bands, output_folder)
        i = interpolate_intensities(i, number_of_bands)
        if args.debug:
            if args.debug:
                debug_save_intensities(i, number_of_bands, output_folder, "_interpolated")
        c = convert_orthophotos_to_reflectance(batch, i)
        del i
        save_orthophotos(batch, c)
        del c
    if args.debug:
        debug_combine_and_plot_intensities(len(orthophoto_paths), number_of_bands, output_folder, panel_properties)
        debug_combine_and_plot_intensities(len(orthophoto_paths), number_of_bands, output_folder, panel_properties,
                                           "_interpolated")


def apriltag_main():
    img_paths = get_apriltag_paths(args.images)
    batches = build_batches_per_band(img_paths)
    number_of_bands = len(batches)


    # TODO: add to args
    tag_size_m = 0.3
    panel_size_m = (0.8, 0.8)

    d = AprilTagDetector()
    d.addFamily(args.family)
    detector_config = get_detector_config()
    d.setConfig(detector_config)

    all_ids = [p["single_tag"] for p in panel_properties]

    # Hack TODO: remove, as panels should not specify family
    d = AprilTagDetector()
    for family in [p["family"] for p in panel_properties]:
        d.addFamily(family)
    d.setConfig(detector_config)

    output_folder = None
    if args.debug:
        output_folder = args.images + "/debug/"
        for p in Path(output_folder + "intensity").glob("*.csv"):
            p.unlink()
        for p in Path(output_folder + "panels").glob("*.tif"):
            p.unlink()

    for (band_index, batch) in enumerate(batches):
        print("Processing batch for band", band_index, "with length", len(batch))
        # --- Run pipeline
        i = extract_intensities_from_apriltags(batch, d, all_ids, panel_size_m, tag_size_m)
        if args.debug:
            debug_save_intensities_single_band(i, band_index, output_folder + "intensity")
        for panel_index, _ in enumerate(panel_properties):
            i[:, panel_index] = interpolate(i[:, panel_index])
        if args.debug:
            debug_save_intensities_single_band(i, band_index, output_folder + "intensity", "_interpolated")
        c = convert_images_to_reflectance(batch, i, band_index)
        del i
        save_images(batch, c)
        del c
    if args.debug:
        number_of_image_per_band = int(len(img_paths) / number_of_bands)
        debug_combine_and_plot_intensities(number_of_image_per_band, number_of_bands, output_folder + "intensity",
                                           panel_properties)
        debug_combine_and_plot_intensities(number_of_image_per_band, number_of_bands, output_folder + "intensity",
                                           panel_properties,
                                           "_interpolated")


if __name__ == '__main__':
    # --- Get input arguments from user
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        prog='ReflectDetect',
        description='Automatically detect reflection calibration panels in images and transform the given images to '
                    'reflectance',
        epilog='If you have any questions, please contact')
    parser.add_argument("detection_type", help="Type of detection ('apriltag', 'geolocation')",
                        type=str)
    parser.add_argument("images", help="Path to the image files", type=str)
    parser.add_argument("panel_properties", help="Path to the property file of the panels", type=str)
    parser.add_argument("--panel_locations", help="Path to the GeoPackage file", type=str, required=False)
    parser.add_argument("--family", help="Name of the apriltag family ('tag25h9', 'tagStandard41h12', ...)", type=str,
                        required=False)
    parser.add_argument("--debug", help="Prints debug images into a /debug/ directory", default=False,
                        action='store_true', required=False)
    args = parser.parse_args()

    panel_properties = load_panel_properties(args.panel_properties)

    detection_type: str = args.detection_type

    # Input Validation
    if detection_type not in ['apriltag', 'geolocation']:
        raise Exception('detection_type ' + detection_type + ' unknown')

    if args.detection_type == 'apriltag':
        for panel_i, panel in enumerate(panel_properties):
            if panel["single_tag"] is None:
                raise Exception(
                    f"Panel {str(panel_i)} Error: Specify the tag id for each panel using the 'single_tag' field in "
                    f"the panel properties file")
            if isinstance(panel["single_tag"], list):
                raise Exception(
                    "Only one tag per panel in apriltag_single mode. Change your panel properties file")
        if args.family is None:
            raise Exception("No family specified globally or for all panels")
        test_detector = AprilTagDetector()
        if not test_detector.addFamily(args.family):
            raise Exception("Family not recognized")

    if args.detection_type == 'geolocation':
        if args.panel_location is None:
            raise Exception("No panel location file specified")

    if detection_type == "apriltag":
        apriltag_main()
    if detection_type == 'geolocation':
        orthophoto_main()
