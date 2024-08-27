import math
from pathlib import Path
from typing import List

import numpy as np
from robotpy_apriltag import AprilTagDetection, AprilTagDetector, AprilTagPoseEstimator
from tifffile import imwrite
from wpimath.geometry import Transform3d

from utils.paths import get_output_path

tag_detection_to_total_width_conversions = {
    "tag16h5": 1.33,
    "tag25h9": 1.22,
    "tag36h11": 1.25,
    "tagCircle21h7": 1.8,
    "tagCircle49h12": 2.2,
    "tagStandard41h12": 1.8,
    "tagStandard52h13": 1.67,
}


def verify_detections(tag, valid_ids: list[int] | None = None) -> bool:
    if valid_ids is None:
        valid_ids = [0, 4, 9]  # ids used by our system by default
    return tag.getId() in valid_ids


def verify_estimate(tag: AprilTagDetection, estimate: Transform3d, valid_ids: list[int], flight_height: float,
                    tolerance=0.1) -> bool:
    return tag.getId() in valid_ids and math.isclose(estimate.z, flight_height, rel_tol=tolerance)


def detect_tags(img, detector: AprilTagDetector, valid_ids: list[int] | None = None):
    tags: List[AprilTagDetection] = detector.detect(img)
    return [tag for tag in tags if verify_detections(tag, valid_ids)]


def pose_estimate_tags(tags: List[AprilTagDetection], config: AprilTagPoseEstimator.Config) -> \
        list[Transform3d]:
    pose_estimator = AprilTagPoseEstimator(config)
    estimates = [pose_estimator.estimate(tag) for tag in tags]
    return estimates  # if verify_estimate(tag, estimate, valid_ids)]


def get_altitude_from_panels(tags: list[AprilTagDetection], path: Path, resolution: tuple[int, int], tag_size: float, ):
    config = get_pose_estimator_config(path, resolution, tag_size)
    estimates = pose_estimate_tags(tags, config)
    return np.mean([estimate.translation().z for estimate in estimates])


def get_pose_estimator_config(path: Path, resolution: tuple[int, int],
                              tag_size: float, ) -> AprilTagPoseEstimator.Config:
    from geolocation_main import get_camera_properties
    from utils.panel import calculate_sensor_size
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
def get_detector_config():
    detector_config = AprilTagDetector.Config()
    detector_config.quadDecimate = 1.0
    detector_config.numThreads = 4
    detector_config.refineEdges = 1.0
    return detector_config

def get_panel(tag: AprilTagDetection, panel_size_pixel: float, image_dimensions: tuple[int, int],
              only_valid_panels=True) -> list[float] | None:
    tag_corners = list(tag.getCorners(tuple([0.0] * 8)))
    tag_corners = np.array(list((zip(tag_corners[::2], tag_corners[1::2]))))

    towards_panel = tag_corners[2] - tag_corners[1]

    tag_detection_size_pixel = np.linalg.norm(towards_panel)
    tag_size = tag_detection_size_pixel * tag_detection_to_total_width_conversions[tag.getFamily()]

    towards_panel = towards_panel / np.linalg.norm(towards_panel)

    center = np.array([tag.getCenter().x, tag.getCenter().y])
    tag_panel_border = center + towards_panel * (tag_size / 2)
    panel_length = towards_panel * panel_size_pixel
    half_panel_length = panel_length / 2
    panel_midpoint_to_corner = [-half_panel_length[1], half_panel_length[0]]

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


def save_images(paths: list[Path], converted_images: list[np.ndarray]) -> None:
    """
    This function saves the converted photos as .tif files into a new "/transformed/" directory in the images folder
    :param paths: List of image paths
    :param converted_images: List of reflectance images
    """
    for path, photo in zip(paths, converted_images):
        if photo is None:
            continue
        output_path = get_output_path(path, "reflectance", "transformed")
        compression_factor = 10000  # convert from 0.1234 to 1234 TODO: Document compression factor
        scaled_to_int = np.array(photo * compression_factor, dtype=np.uint8)
        imwrite(output_path, scaled_to_int)
