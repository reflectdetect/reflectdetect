import argparse
import logging
from pathlib import Path

from src.detector.dummy.DummyDetector import DummyDetector
from src.extractor.dummy.DummyExtractor import DummyExtractor
from src.transformer.dummy.DummyTransformer import DummyTransformer

logger = logging.getLogger(__name__)


def run_detection(path):
    detector = DummyDetector()
    extractor = DummyExtractor()
    transformer = DummyTransformer()

    def pipeline(p: str):
        logger.info(f"Transforming image {p} to reflectance")
        detection_results_path = detector.detect(p)
        extraction_results_path = extractor.extract(p, detection_results_path)
        transformed_image_path = transformer.transform(p, extraction_results_path)

        return transformed_image_path, extraction_results_path, detection_results_path

    template_path = (Path.cwd() / path).resolve()

    file_endings = (".jpg", ".png", ".tif")

    if template_path.is_dir():
        images_path = (template_path / 'Images/seq1').resolve()
        annotations_path = (template_path / 'annotations').resolve()
        results = []
        files = []

        for e in file_endings:
            files.extend(images_path.glob("*" + e))
        for filename in files:
            results.append((filename, *(pipeline(images_path.joinpath(filename)))))
        return results
    else:
        if template_path.endswith(file_endings):
            return list((template_path, pipeline(template_path)))
        else:
            return []


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        prog='ReflectDetect',
        description='Automatically detect reflection calibration panels in images and transform the given images to '
                    'reflectance',
        epilog='If you have any questions, please contact')
    parser.add_argument("path", help="Path to the image file or image files directory", type=str)
    args = parser.parse_args()

    run_detection(args.path)
