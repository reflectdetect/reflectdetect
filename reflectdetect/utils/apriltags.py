import contextlib
import io
import math
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from rich.progress import Progress
from robotpy_apriltag import AprilTagDetection, AprilTagDetector, AprilTagPoseEstimator
from tifffile import imwrite
from wpimath.geometry import Transform3d

from reflectdetect.utils.debug import ProgressBar
from reflectdetect.utils.exif import get_camera_properties
from reflectdetect.utils.panel import calculate_sensor_size
from reflectdetect.utils.paths import get_output_path
from reflectdetect.utils.thread import run_in_thread

# The robotpy_apriltag.AprilTagDetector returns the inner apriltag square coordinates and not the whole apriltag area.
# Therefore the detection area has to be converted to the full apriltag area to get the accurate distance
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


def verify_detections(tag: AprilTagDetection, valid_ids: list[int] | None = None) -> bool:
    if valid_ids is None:
        valid_ids = [0, 4, 9]  # ids used by our system by default
    return tag.getId() in valid_ids


def verify_estimate(tag: AprilTagDetection, estimate: Transform3d, valid_ids: list[int], flight_height: float,
                    tolerance: float = 0.1) -> bool:
    return tag.getId() in valid_ids and math.isclose(estimate.z, flight_height, rel_tol=tolerance)


def detect_tags(img: NDArray[np.float64], detector: AprilTagDetector, valid_ids: list[int] | None = None) -> list[
    AprilTagDetection]:
    tags: list[AprilTagDetection] = run_in_thread(detector.detect, True, img)  # type: ignore
    return [tag for tag in tags if verify_detections(tag, valid_ids)]


def pose_estimate_tags(tags: list[AprilTagDetection], config: AprilTagPoseEstimator.Config) -> \
        list[Transform3d]:
    pose_estimator = AprilTagPoseEstimator(config)
    estimates: list[Transform3d] = [run_in_thread(pose_estimator.estimate, False, tag) for tag in
                                    tags]  # type: ignore
    return estimates


def get_altitude_from_panels(tags: list[AprilTagDetection], path: Path, resolution: tuple[int, int],
                             tag_size: float) -> float:
    config = get_pose_estimator_config(path, resolution, tag_size)
    with contextlib.redirect_stdout(io.StringIO()):
        estimates = pose_estimate_tags(tags, config)
    return float(np.mean([estimate.translation().z for estimate in estimates]))


def get_pose_estimator_config(path: Path, resolution: tuple[int, int],
                              tag_size: float, ) -> AprilTagPoseEstimator.Config:
    focal_length_mm, focal_plane_x_res, focal_plane_y_res, focal_plane_resolution_unit = get_camera_properties(path)
    horizontal_focal_length_pixels = focal_length_mm * focal_plane_x_res
    vertical_focal_length_pixels = focal_length_mm * focal_plane_y_res

    sensor_width_mm, sensor_height_mm = calculate_sensor_size(resolution, focal_plane_x_res, focal_plane_y_res,
                                                              focal_plane_resolution_unit)
    sensor_width_pixel = sensor_width_mm * focal_plane_x_res
    sensor_height_pixel = sensor_height_mm * focal_plane_y_res
    horizontal_focal_center_pixels = sensor_width_pixel / 2
    vertical_focal_center_pixels = sensor_height_pixel / 2

    return AprilTagPoseEstimator.Config(tag_size, horizontal_focal_length_pixels,
                                        vertical_focal_length_pixels,
                                        horizontal_focal_center_pixels, vertical_focal_center_pixels)


def get_detector_config() -> AprilTagDetector.Config:
    detector_config = AprilTagDetector.Config()
    detector_config.quadDecimate = 1.0
    detector_config.numThreads = 4
    detector_config.refineEdges = True
    return detector_config


def get_panel(tag: AprilTagDetection, panel_size_pixel: tuple[int, int], image_dimensions: tuple[int, int],
              tag_smudge_factor: float,
              only_valid_panels: bool = True) -> list[float] | None:
    tag_corners = np.array(list(tag.getCorners((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))))
    tag_corners = np.array(list((zip(tag_corners[::2], tag_corners[1::2]))))

    towards_panel = tag_corners[2] - tag_corners[1]

    tag_detection_size_pixel = np.linalg.norm(towards_panel)
    tag_size = tag_detection_size_pixel * tag_detection_to_total_width_conversions[tag.getFamily()]

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


def save_images(dataset: Path, paths: list[Path], converted_images: list[NDArray[np.float64] | None],
                progress: Progress | None = None) -> None:
    """
    This function saves the converted photos as .tif files into a new "/transformed/" directory in the images folder
    :param paths: list of image paths
    :param converted_images: list of reflectance images
    """
    with ProgressBar(progress, description="Saving images", total=len(paths)) as pb:
        for path, photo in zip(paths, converted_images):
            if photo is None:
                pb.update()
                continue
            output_path = get_output_path(dataset, path, "reflectance.tif", "transformed")
            compression_factor = 10000  # convert from 0.1234 to 1234 TODO: Document compression factor
            scaled_to_int = np.array(photo * compression_factor, dtype=np.uint8)
            imwrite(output_path, scaled_to_int)
            pb.update()
