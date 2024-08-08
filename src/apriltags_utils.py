import math
from typing import List

import numpy as np
import shapely
from rasterio.features import rasterize
from robotpy_apriltag import AprilTagDetection, AprilTagDetector, AprilTagPoseEstimator
from wpimath.geometry import Transform3d


def verify_detection(tag: AprilTagDetection, valid_ids: list[int]) -> bool:
    return tag.getId() in valid_ids


def verify_estimate(tag: AprilTagDetection, estimate: Transform3d, valid_ids: list[int], flight_height: float,
                    tolerance=0.1) -> bool:
    return tag.getId() in valid_ids and math.isclose(estimate.z, flight_height, rel_tol=tolerance)


def detect_tags(img, detector: AprilTagDetector, valid_ids: list[int]):
    tags: List[AprilTagDetection] = detector.detect(img)
    return [tag for tag in tags if verify_detection(tag, valid_ids)]


def pose_estimate_tags(img, detector: AprilTagDetector, config: AprilTagPoseEstimator.Config, valid_ids: list[int]) -> \
        list[tuple[AprilTagDetection, Transform3d]]:
    pose_estimator = AprilTagPoseEstimator(config)

    # AprilTagPoseEstimator.Config(tag_size, horizontal_focal_length_pixels, vertical_focal_length_pixels,
    #                                      horizontal_focal_center_pixels, vertical_focal_center_pixels)

    tags: List[AprilTagDetection] = detector.detect(img)
    estimates = [(tag, pose_estimator.estimate(tag)) for tag in tags]

    return [estimate for estimate in estimates if verify_estimate(estimate[0], estimate[1], valid_ids)]


def extract_edges(img, tags: list[AprilTagDetection]):
    polygon = shapely.Polygon([tag.center() for tag in tags]).convex_hull()
    mask = rasterize([polygon], out_shape=img.shape)
    # mean the radiance values to get a radiance value for the detection
    mean = np.ma.array(img, mask=~(mask.astype(np.bool_))).mean()
    return mean


def extract_single(img, tag: (AprilTagDetection, Transform3d), tag_size: float):
    raise NotImplementedError
