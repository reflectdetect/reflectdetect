import math
from pathlib import Path
from typing import List

import numpy as np
import rasterio
from rasterio.io import DatasetWriter
from robotpy_apriltag import AprilTagDetection, AprilTagDetector, AprilTagPoseEstimator
from wpimath.geometry import Transform3d

from utils.paths import get_output_path

# TODO: Fill out for all families
tag_detection_to_total_width_conversions = {
    "tagCircle21h7": 0.5 / 0.28,
    "tag25h9": 0.44 / 0.36,
    "tagCircle49h12": 0.56 / 0.25,
    "tagStandard52h13": 0.56 / 0.31
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
        list[tuple[AprilTagDetection, Transform3d]]:
    pose_estimator = AprilTagPoseEstimator(config)
    estimates = [(tag, pose_estimator.estimate(tag)) for tag in tags]
    return estimates  # [estimate for (tag, estimate) in estimates]  # if verify_estimate(tag, estimate, valid_ids)]


def get_altitude_from_panels(tags: list[AprilTagDetection], config: AprilTagPoseEstimator.Config):
    estimates = pose_estimate_tags(tags, config)
    return np.mean([estimate.translation().z for (tag, estimate) in estimates])


def get_panel_st(tag, panel_size_pixel: float) -> tuple[
                                                      AprilTagDetection, list[float]] | None:
    # Panel size calculation
    # assumes square panels
    corners = list(tag.getCorners(tuple([0.0] * 8)))
    corners = np.array(list((zip(corners[::2], corners[1::2]))))
    towards_panel = corners[2] - corners[1]
    tag_detection_size_pixel = np.linalg.norm(towards_panel)
    tag_size = tag_detection_size_pixel * tag_detection_to_total_width_conversions[tag.getFamily()]
    towards_panel = towards_panel / np.linalg.norm(towards_panel)
    center = np.array([tag.getCenter().x, tag.getCenter().y])
    tag_panel_border = center + towards_panel * (tag_size / 2)
    panel_length = towards_panel * panel_size_pixel
    half_panel_length = panel_length / 2
    panel_midpoint_to_edge = [-half_panel_length[1], half_panel_length[0]]
    edge_a = tag_panel_border + panel_midpoint_to_edge
    edge_b = tag_panel_border - panel_midpoint_to_edge
    edge_c = tag_panel_border + panel_length - panel_midpoint_to_edge
    edge_d = tag_panel_border + panel_length + panel_midpoint_to_edge
    return tag, [edge_a, edge_b, edge_c, edge_d]


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
        with rasterio.open(path) as original:
            meta = original.meta
        meta.update(
            dtype=rasterio.float32,
            transform=rasterio.Affine.identity(),
        )
        dst: DatasetWriter
        with rasterio.open(output_path, 'w', **meta) as dst:
            dst.write_band(1, photo)
