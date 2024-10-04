import contextlib
import io
import re
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from exiftool import ExifToolHelper
from numpy.typing import NDArray
from rich.progress import Progress
from robotpy_apriltag import AprilTagDetection, AprilTagDetector, AprilTagPoseEstimator
from tifffile import imwrite
from wpimath.geometry import Transform3d

from reflectdetect.constants import COMPRESSION_FACTOR, CONVERTED_FILE_ENDING
from reflectdetect.utils.debug import ProgressBar
from reflectdetect.utils.exif import get_camera_properties
from reflectdetect.utils.panel import calculate_sensor_size
from reflectdetect.utils.paths import default
from reflectdetect.utils.paths import get_output_path
from reflectdetect.utils.thread import run_in_thread

# The robotpy_apriltag.AprilTagDetector returns the inner apriltag square coordinates and not the whole apriltag area.
# Therefore, the detection area has to be converted to the full apriltag area to get the accurate distance
# from the center of the tag to the edge of the panel
tag_detection_to_total_width_conversions = {
    "tag16h5": 1.33,
    "tag25h9": 1.22,
    "tag36h11": 1.25,
    "tagCircle21h7": 1.8,
    "tagCircle49h12": 2.2,
    "tagStandard41h12": 1.8,
    "tagStandard52h13": 1.67,
}


def verify_detections(tag: AprilTagDetection, valid_ids: list[int]) -> bool:
    """
    Check if the tag id is valid
    :param tag: the apriltag
    :param valid_ids: ids that are printed on the apriltags
    :return: whether the tag has an expected id
    """
    return tag.getId() in valid_ids


def detect_tags(
        img: NDArray[Any], detector: AprilTagDetector, valid_ids: list[int]
) -> list[AprilTagDetection]:
    """
    Detect apriltags in an image
    :param img: the image to detect apriltags in
    :param detector: the detector to use for detecting
    :param valid_ids: the ids the apriltags can have, to discard wrong detections
    :return: the detected apriltags
    """
    tags: list[AprilTagDetection] = run_in_thread(detector.detect, True, img)  # type: ignore
    return [tag for tag in tags if verify_detections(tag, valid_ids)]


def pose_estimate_tags(
        tags: list[AprilTagDetection], config: AprilTagPoseEstimator.Config
) -> list[Transform3d]:
    """
    Create a pose estimate for each given tag
    :param tags: apriltags to pose estimate
    :param config: configuration for the pose estimator
    :return: the pose estimate
    """
    pose_estimator = AprilTagPoseEstimator(config)
    with warnings.catch_warnings():
        # Ignore PoseEstimation Warning
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        estimates: list[Transform3d] = [
            run_in_thread(pose_estimator.estimate, True, tag)  # type: ignore
            for tag in tags
        ]
    return estimates


def get_altitude_from_tags(
        exiftool: ExifToolHelper,
        tags: list[AprilTagDetection],
        path: Path,
        resolution: tuple[int, int],
        tag_size: float,
) -> float:
    """
    Approximate the altitude of the drone taking the image by pose estimating the tags in the image
    and meaning the height
    :param tags: the tags to pose estimate
    :param path: path to the image
    :param resolution: resolution of the image
    :param tag_size: size of the tags (detection area, not total area)
    :return:  the approximate height of the drone taking the image
    """
    config = get_pose_estimator_config(exiftool, path, resolution, tag_size)
    with contextlib.redirect_stdout(io.StringIO()):
        estimates = pose_estimate_tags(tags, config)
    return float(np.mean([estimate.translation().z for estimate in estimates]))


def get_pose_estimator_config(
        exiftool: ExifToolHelper,
        path: Path,
        resolution: tuple[int, int],
        tag_size: float,
) -> AprilTagPoseEstimator.Config:
    """
    Calculate the pose estimator configuration for the given image.
    Camera properties will be extracted from the exif data
    :param path: path to the image
    :param resolution: resolution of the image
    :param tag_size: size of the apriltag in meters (detection area, not total area)
    :return:
    """
    (
        focal_length_mm,
        focal_plane_x_res,
        focal_plane_y_res,
        focal_plane_resolution_unit,
    ) = get_camera_properties(exiftool, path)
    horizontal_focal_length_pixels = focal_length_mm * focal_plane_x_res
    vertical_focal_length_pixels = focal_length_mm * focal_plane_y_res

    sensor_width_mm, sensor_height_mm = calculate_sensor_size(
        resolution, focal_plane_x_res, focal_plane_y_res, focal_plane_resolution_unit
    )
    sensor_width_pixel = sensor_width_mm * focal_plane_x_res
    sensor_height_pixel = sensor_height_mm * focal_plane_y_res
    horizontal_focal_center_pixels = sensor_width_pixel / 2
    vertical_focal_center_pixels = sensor_height_pixel / 2

    return AprilTagPoseEstimator.Config(
        tag_size,
        horizontal_focal_length_pixels,
        vertical_focal_length_pixels,
        horizontal_focal_center_pixels,
        vertical_focal_center_pixels,
    )


def get_detector_config() -> AprilTagDetector.Config:
    """
    Get the default apriltag detector configuration. QuadDecimate is set to 1 for best performance even with small tags
    :return:  default apriltag detector configuration
    """
    detector_config = AprilTagDetector.Config()
    detector_config.quadDecimate = 1.0
    detector_config.numThreads = 4
    detector_config.refineEdges = True
    return detector_config


def build_batches_per_band(paths: list[Path]) -> list[list[Path]]:
    """
    Create a batch of images for each band
    :param paths: image paths
    :return: list of batches with each batch containing all images of a given band
    """
    # TODO also add visibility batching
    batches: list[list[Path]] = []
    # Regular expression to match file names and capture the base and suffix
    pattern = re.compile(fr".*_(\d+)_{CONVERTED_FILE_ENDING}$")  # TODO better path parsing generalization

    for image_path in paths:
        match = pattern.match(image_path.name)
        if not match:
            raise Exception("Could not extract band index from filename")
        band_index = int(match.group(1)) - 1
        if band_index > len(batches) or band_index < 0:
            raise Exception("Problem with the sorting of the paths or regex")
        if band_index == len(batches):
            batches.append([])
        batches[band_index].append(image_path)
    return batches


def get_panel(
        tag: AprilTagDetection,
        panel_size_pixel: tuple[int, int],
        image_dimensions: tuple[int, int],
        tag_smudge_factor: float,
        tag_direction: str,
        only_valid_panels: bool = True,
) -> list[tuple[float, float]] | None:
    """
    Get the corners of a panel based on a apriltag to its side
    :param tag: the apriltag that was detected in the image
    :param panel_size_pixel: the size of the panel in the image
    :param image_dimensions: the dimensions of the image
    :param tag_smudge_factor: factor to change the distance of the apriltag to the panel,
    it for example on of the tags was put further away from the panel as planned
    :param tag_direction: direction of the panel relative to the tag (up, down, left, right)
    :param only_valid_panels: whether to filter out panels that have corners outside the image dimensions
    :return: the 4 corners of the panel in tuples of x y image coordinates
    """
    tag_corners = np.array(
        list(tag.getCorners((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)))
    )
    tag_corners = np.array(list((zip(tag_corners[::2], tag_corners[1::2]))))

    if tag_direction == "up":
        towards_panel = tag_corners[2] - tag_corners[1]
    if tag_direction == "down":
        towards_panel = tag_corners[1] - tag_corners[2]
    if tag_direction == "left":
        towards_panel = tag_corners[1] - tag_corners[0]
    if tag_direction == "right":
        towards_panel = tag_corners[0] - tag_corners[1]

    tag_detection_size_pixel = np.linalg.norm(towards_panel)
    tag_size = (
            tag_detection_size_pixel
            * tag_detection_to_total_width_conversions[tag.getFamily()]
    )

    towards_panel = towards_panel / np.linalg.norm(towards_panel)

    center = np.array([tag.getCenter().x, tag.getCenter().y])
    tag_panel_border = center + towards_panel * (tag_size / 2) * tag_smudge_factor
    panel_length = towards_panel * panel_size_pixel[1]
    panel_width = towards_panel * panel_size_pixel[0]
    half_panel_width = panel_width / 2
    panel_midpoint_to_corner = [-half_panel_width[1], half_panel_width[0]]

    corner_a = tag_panel_border + panel_midpoint_to_corner
    corner_b = tag_panel_border - panel_midpoint_to_corner
    corner_c = tag_panel_border + panel_length - panel_midpoint_to_corner
    corner_d = tag_panel_border + panel_length + panel_midpoint_to_corner
    corners = [corner_a, corner_b, corner_c, corner_d]

    if only_valid_panels:
        for corner in corners:
            if corner[0] < 0 or corner[1] < 0:
                return None
            if corner[0] > image_dimensions[0] or corner[1] > image_dimensions[1]:
                return None

    return corners


def save_images(
        exiftool: ExifToolHelper,
        dataset: Path,
        paths: list[Path],
        converted_images: list[NDArray[np.float64] | None],
        progress: Progress | None = None,
        output_folder: str | None = None,
        ending: str | None = None,
) -> None:
    """
    This function saves the converted images as .tif files into a new "/transformed/" directory in the images folder
    :param dataset: path to the dataset to save the images to
    :param progress: optional progress bar
    :param paths: list of image paths
    :param converted_images: list of reflectance images
    :param output_folder: Overwrite for the transformed directory name
    :param ending: Overwrite for the file ending
    """
    with ProgressBar(progress, description="Saving images", total=len(paths)) as pb:
        path: Path
        for path, image in zip(paths, converted_images):
            if image is None:
                pb.update()
                continue
            output_path = get_output_path(
                dataset, path, default(ending, "reflectance.tif"), default(output_folder, "transformed")
            )
            image[image < 0] = 0
            scaled_to_int = np.array(image * COMPRESSION_FACTOR, dtype=np.uint16)
            imwrite(output_path, scaled_to_int)
            # Copy the exifdata from the original image to the new one
            run_in_thread(exiftool.execute, True, b"-overwrite_original", b"-tagsFromFile", path.as_posix(),
                          output_path.as_posix())
            pb.update()
