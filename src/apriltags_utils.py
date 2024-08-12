import math
from typing import List

import numpy as np
import shapely
from rasterio.features import rasterize
from robotpy_apriltag import AprilTagDetection, AprilTagDetector, AprilTagPoseEstimator
from wpimath.geometry import Transform3d

from detector.naive.NaiveDetector import calculate_panel_size_in_pixels

#TODO: Fill out for all families
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


def pose_estimate_tags(img, detector: AprilTagDetector, config: AprilTagPoseEstimator.Config,
                       valid_ids: list[int] | None = None) -> \
        list[tuple[AprilTagDetection, Transform3d]]:
    pose_estimator = AprilTagPoseEstimator(config)
    tags: List[AprilTagDetection] = detector.detect(img)
    estimates = [(tag, pose_estimator.estimate(tag)) for tag in tags if verify_detections(tag, valid_ids)]

    return [estimate for (tag, estimate) in estimates]  # if verify_estimate(tag, estimate, valid_ids)]


def get_altitude_from_panels(img, detector, config: AprilTagPoseEstimator.Config, valid_ids=None, ):
    estimates = pose_estimate_tags(img, detector, config, valid_ids)
    return np.mean([estimate.translation().z for (tag, estimate) in estimates])


def get_panels_st(img, detector, sensor_size_mm: float, focal_length_mm: float, panel_size_m: float,
                  config: AprilTagPoseEstimator.Config, valid_ids=None, ):
    tags = detect_tags(img, detector, valid_ids)
    panels = []

    # Panel size calculation
    altitude = get_altitude_from_panels(img, detector, config, valid_ids)
    resolution = (len(img[0]), len(img))
    panel_size_pixel = calculate_panel_size_in_pixels(altitude, resolution, sensor_size_mm, focal_length_mm,
                                                      (panel_size_m, panel_size_m))  # assumes square panels
    for tag in tags:
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
        panels.append((tag, (edge_a, edge_b, edge_c, edge_d)))
    return panels
