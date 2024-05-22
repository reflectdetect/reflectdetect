import csv
import json
import os
from datetime import datetime

import shapely.geometry as sg

from src.detector.dummy.DummyDetector import DummyDetector
from src.extractor.naive.Extractor import Extractor
from src.main import run_pipeline_for_each_image
from src.transformer.naive.Transformer import Transformer
from src.utils.paths import get_image_id, get_image_band


def test_pytest():
    assert True


def test_main_performance():
    dataset_path = "data/example/YOLO_OBB_Dataset"
    detector = DummyDetector()
    extractor = Extractor("reflectance_panel_example_data.json")
    transformer = Transformer()
    results = run_pipeline_for_each_image(detector, extractor, transformer, dataset_path)
    assert len(results) == 5
    overall_metrics = {"dataset": dataset_path,
                       "detector": detector.get_name(),
                       "extractor": extractor.get_name(),
                       "transformer": transformer.get_name()
                       }
    last_image_id = None
    metrics = {}
    for image_path, transformed_image_path, extraction_path, detection_path in results:
        if get_image_id(image_path) != last_image_id:
            if last_image_id is not None:
                overall_metrics[str(last_image_id)] = metrics
            metrics = {}
        metrics[str(get_image_band(image_path))] = get_metrics(dataset_path, image_path, transformed_image_path, extraction_path,
                                                                   detection_path)
        last_image_id = get_image_id(image_path)
    overall_metrics[str(last_image_id)] = metrics
    save_metrics(overall_metrics)


def get_metrics(dataset_path, image_path, transformed_image_path, extraction_path, detection_path):
    with open(dataset_path + "/annotations/seq1.csv") as annotations_file:
        reader = csv.DictReader(annotations_file, delimiter=',')
        annotations = [row for row in reader]
        identifier = get_image_id(image_path)
        band = get_image_band(image_path)
        ground_truth_annotations = [row for row in annotations if int(row["id"]) == int(identifier) and int(row["band"]) == band]

        metrics = []
        # Detection_metric
        with open(detection_path) as f:
            detections = json.load(f)
            for detection in detections:
                for gt in ground_truth_annotations:
                    gt_list = list(gt.values())[2:]
                    gt_list = list(zip(gt_list, gt_list[1:]))[::2]
                    detection_list = detection[2:]
                    detection_list = list(zip(detection_list, detection_list[1:]))[::2]
                    metrics.append(calculate_overlap_metrics(detection_list, gt_list))
        return metrics


def calculate_overlap_metrics(rect1, rect2):
    """
    Calculates overlap metrics between two rotated rectangles.

    Args:
        rect1: A list of 4 points representing the vertices of the first rectangle.
        rect2: A list of 4 points representing the vertices of the second rectangle.

    Returns:
        A dictionary containing the following metrics:
            * area: The area of the overlap.
            * overlap_percentage_rect1: The percentage of rect1's area that is overlapped.
            * overlap_percentage_rect2: The percentage of rect2's area that is overlapped.
            * iou: The Intersection over Union, a common metric for object detection.
    """

    poly1 = sg.Polygon(rect1)
    poly2 = sg.Polygon(rect2)

    intersection_poly = poly1.intersection(poly2)
    overlap_area = intersection_poly.area if not intersection_poly.is_empty else 0

    area_rect1 = poly1.area
    area_rect2 = poly2.area

    union_area = area_rect1 + area_rect2 - overlap_area

    metrics = {
        "area": overlap_area,
        "overlap_percentage_rect1": (overlap_area / area_rect1) * 100 if area_rect1 > 0 else 0,
        "overlap_percentage_rect2": (overlap_area / area_rect2) * 100 if area_rect2 > 0 else 0,
        "iou": (overlap_area / union_area) * 100 if union_area > 0 else 0,
    }

    return metrics


def save_metrics(metrics, filename_prefix="metrics/metrics"):
    """
    Saves overlap metrics to a timestamped CSV file.

    Args:
        metrics: A dictionary of metrics (output of calculate_overlap_metrics).
        filename_prefix (optional): A prefix for the filename. Defaults to "overlap_metrics".
    """

    # Check if running in GitHub Actions
    is_github_actions = os.getenv("GITHUB_ACTIONS") == "true"

    # Generate timestamped filename with postfix if applicable
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename_prefix}_{timestamp}{'_github' if is_github_actions else ''}.json"

    # Write to CSV file
    with open(filename, "w") as jsonfile:
        json.dump(metrics, jsonfile, indent=4)  # Use indent for pretty formatting

    print(f"Metrics saved to {filename}")
