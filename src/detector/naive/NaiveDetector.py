import json
import math

import cv2
import numpy as np
from matplotlib import pyplot as plt

from src.detector.BaseDetector import BaseDetector
from src.utils.panel_utils import get_panel_factors_for_band
from src.utils.paths import get_image_band


def load_image(path: str):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    transformed_image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
    return transformed_image


class NaiveDetector(BaseDetector):

    def __init__(self, path_to_panel_data: str):
        with open(path_to_panel_data) as f:
            data = json.load(f)
            self.panels = data
        # TODO(add function to read from config file)
        # Given/known Camera parameters
        self.aspect_ratio_deviation = .8  # Threshold for aspect ratio deviation
        self.min_solidity = 0.60  # Set a threshold for solidity to filter out non-homogeneous areas
        self.area_deviation_smaller = 0.5
        self.area_deviation_larger = 1.2
        self.altitude = 16.3145  # Altitude in meters
        self.resolution = (1456, 1088)  # Image resolution (width, height) in pixels
        self.sensor_size_mm = 6.3  # Sensor diagonal in millimeters
        self.focal_length_mm = 5.5  # Focal length in millimeters
        self.physical_panel_size = (1, 1)  # Physical size of the object in meters (width, height)

    def preprocess_image(self, image, band, blur_kernel_size=41):
        # variance filter
        variance = filter_by_variance(image)
        blur = cv2.medianBlur(variance, blur_kernel_size)
        ret, thresh1 = cv2.threshold(blur, 1, 255, cv2.THRESH_BINARY_INV)

        images = [thresh1]
        panels = get_panel_factors_for_band(self.panels, band)
        for panel in panels:
            images.append(
                np.logical_and(image >= 255 * (panel - 0.05), image <= 255 * (panel + 0.05)).astype(np.uint8) * 255)
            images.append(
                (~np.logical_and(image >= 255 * (panel - 0.05), image <= 255 * (panel + 0.05))).astype(np.uint8) * 255)

        return images

    def detect(self, image_path: str) -> str:
        image = load_image(image_path)
        panel_size = calculate_panel_size_in_pixels(
            self.altitude,
            self.resolution,
            self.sensor_size_mm,
            self.focal_length_mm,
            self.physical_panel_size
        )
        kernel_size = (int(panel_size[0] / 8) * 2) + 1  # use 1/4th of the panel size as a blur kernel

        band = get_image_band(image_path)
        images = self.preprocess_image(image, band, blur_kernel_size=kernel_size)

        boxes = []
        for i in images:
            found_boxes = self.get_bound_boxes(i, panel_size)
            boxes = boxes + found_boxes

        # TODO: combine near boxes
        bounding_boxes = self.combine_bounding_boxes(boxes, panel_size[0]/2)

        show_image_with_bboxes(image, [np.array(bbox["coordinates"]) for bbox in bounding_boxes])
        # Save bounding box coordinates to a JSON file
        with open("bounding_boxes.json", "w") as f:
            json.dump(bounding_boxes, f, indent=4)
        return "bounding_boxes.json"

    def convert_contours_to_bboxes(self, filtered_contours, panel_size):
        # Expected aspect ratio based on physical panel size
        expected_aspect_ratio = self.physical_panel_size[0] / self.physical_panel_size[1]
        spacing = panel_size[0] * 0.2  # Amount of spacing to add between the resulting boxes in pixels
        # List to hold bounding box coordinates
        bounding_boxes = []
        for contour in filtered_contours:
            # Get the minimum area rectangle for the contour
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)  # Get the four points of the rectangle
            box = np.intp(box)  # Convert to integer values # TODO(Might not be necessary)

            # Split bounding box if necessary and space apart
            boxes = split_bounding_box(box, expected_aspect_ratio, self.aspect_ratio_deviation, spacing)
            for bbox in boxes:
                # Save the bounding box coordinates
                bounding_box = {
                    "contour_index": len(bounding_boxes),  # Index of the contour
                    "coordinates": bbox.tolist()  # Convert numpy array to list
                }
                bounding_boxes.append(bounding_box)
        return bounding_boxes

    def find_contours(self, image, panel_size):
        # Find contours in the binary mask
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Filter contours based on size (e.g., minimum area) and solidity
        panel_area = panel_size[0] * panel_size[1]
        min_contour_area = panel_area - panel_area * self.area_deviation_smaller
        max_contour_area = panel_area + panel_area * self.area_deviation_larger
        filtered_contours = [c for c in contours if
                             min_contour_area <= cv2.contourArea(c) <= max_contour_area
                             and contour_solidity_by_rect(c) > self.min_solidity
                             ]
        return filtered_contours

    def get_bound_boxes(self, image, panel_size):
        filtered_contours = self.find_contours(image, panel_size)
        bounding_boxes = self.convert_contours_to_bboxes(filtered_contours, panel_size)
        return bounding_boxes

    def combine_bounding_boxes(self, boxes, closeness: int):
        combined_boxes = []
        for index, box in enumerate(boxes):
            for box2 in boxes[index + 1:]:
                center = np.array(box["coordinates"]).mean(0)
                center2 = np.array(box2["coordinates"]).mean(0)
                if math.dist(center, center2) < closeness:
                    break
            else:
                combined_boxes.append(box)
        return combined_boxes


def show_image_with_bboxes(image, boxes):
    i = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(i, boxes, -1, (255, 0, 0), 3)  # Draw the rectangle
    plt.imshow(i)
    plt.show()


def contour_solidity(contour):
    area = cv2.contourArea(contour)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area == 0:
        return 0
    return float(area) / hull_area


def contour_solidity_by_rect(contour):
    area = cv2.contourArea(contour)
    rect = cv2.minAreaRect(contour)
    rect_area = cv2.contourArea(cv2.boxPoints(rect))
    if rect_area == 0:
        return 0
    return float(area) / rect_area


def split_bounding_box(box, expected_aspect_ratio, threshold=0.1, spacing=10):
    """
    Split the bounding box if its aspect ratio deviates significantly from the expected aspect ratio,
    and space apart the resulting boxes by a specified amount.

    Parameters:
        box (ndarray): Array of bounding box points.
        expected_aspect_ratio (float): The expected aspect ratio of the panels.
        threshold (float): The acceptable threshold for aspect ratio deviation.
        spacing (int): The amount of spacing to add between the resulting boxes.

    Returns:
        list: A list of bounding boxes (ndarray).
    """
    rect = cv2.minAreaRect(box)
    width, height = rect[1]

    if width == 0 or height == 0:
        return [box]

    # Calculate the aspect ratio of the bounding box
    aspect_ratio = width / height if width > height else height / width

    # Compare aspect ratio with expected aspect ratio
    if abs(aspect_ratio - expected_aspect_ratio) / expected_aspect_ratio > threshold:
        # Determine if width or height should be split
        if width > height:
            new_width = width / 2
            size = (new_width, height)
            offset = (spacing / 2) * np.array([np.cos(np.deg2rad(rect[2])), np.sin(np.deg2rad(rect[2]))])
            center1 = (rect[0][0] - new_width / 2 * np.cos(np.deg2rad(rect[2])),
                       rect[0][1] - new_width / 2 * np.sin(np.deg2rad(rect[2]))) - offset
            center2 = (rect[0][0] + new_width / 2 * np.cos(np.deg2rad(rect[2])),
                       rect[0][1] + new_width / 2 * np.sin(np.deg2rad(rect[2]))) + offset
        else:
            new_height = height / 2
            size = (width, new_height)
            offset = (spacing / 2) * np.array([np.sin(np.deg2rad(rect[2])), -np.cos(np.deg2rad(rect[2]))])
            center1 = (rect[0][0] - new_height / 2 * np.sin(np.deg2rad(rect[2])),
                       rect[0][1] + new_height / 2 * np.cos(np.deg2rad(rect[2]))) - offset
            center2 = (rect[0][0] + new_height / 2 * np.sin(np.deg2rad(rect[2])),
                       rect[0][1] - new_height / 2 * np.cos(np.deg2rad(rect[2]))) + offset

        # Create two new bounding boxes
        box1 = cv2.boxPoints((center1, size, rect[2]))
        box2 = cv2.boxPoints((center2, size, rect[2]))

        return [np.intp(box1), np.intp(box2)]

    return [box]


def calculate_panel_size_in_pixels(altitude, resolution, sensor_size_mm, focal_length_mm,
                                   physical_panel_size):
    # TODO(Sanitycheck function)
    """
    Calculate the expected size of an object in pixels based on camera parameters and object physical size.

    Parameters:
        altitude (float): Altitude in meters.
        resolution (tuple): Image resolution (width, height) in pixels.
        sensor_size_mm (float): Sensor diagonal in millimeters.
        focal_length_m (float): Focal length in millimeters.
        physical_panel_size (tuple): Physical size of the object in meters (width, height).

    Returns:
        tuple: Expected width and height of the object in pixels.
    """
    # Convert sensor diagonal to meters
    sensor_diagonal = sensor_size_mm / 1000  # Convert mm to m
    focal_length = focal_length_mm / 1000

    # Calculate horizontal and vertical Field of View (FoV)
    fov_horizontal = 2 * math.atan(
        (sensor_diagonal / (2 * math.sqrt(1 + (resolution[0] / resolution[1]) ** 2))) / focal_length)
    fov_vertical = 2 * math.atan(
        (sensor_diagonal / (2 * math.sqrt(1 + (resolution[1] / resolution[0]) ** 2))) / focal_length)

    # Calculate scale in pixels per meter
    scale_pixels_per_meter = resolution[1] / (altitude * math.tan(fov_vertical / 2))

    # Calculate expected panel size in pixels
    panel_width_pixels = np.intp(physical_panel_size[0] * scale_pixels_per_meter)
    panel_height_pixels = np.intp(physical_panel_size[1] * scale_pixels_per_meter)

    return panel_width_pixels, panel_height_pixels


def filter_by_variance(image, blur_kernel_size=21):
    """
    Apply a variance filter to an image to enhance features.

    Parameters:
        image (numpy.ndarray): Input image.
        blur_kernel_size (int): Size of the Gaussian blur kernel.

    Returns:
        numpy.ndarray: Filtered image.
    """

    blur_non = cv2.GaussianBlur(image, (blur_kernel_size, blur_kernel_size), 2)
    for i in range(10):
        blur_non = cv2.GaussianBlur(blur_non, (blur_kernel_size, blur_kernel_size), 2)
    last_blur_non = cv2.GaussianBlur(image, (blur_kernel_size, blur_kernel_size), 2)
    variance_map = (last_blur_non - blur_non)
    return cv2.normalize(variance_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
