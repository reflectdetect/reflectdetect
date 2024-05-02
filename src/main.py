import argparse
import logging
import os.path
from pathlib import Path

from src.detector.dummy.DummyDetector import DummyDetector
from src.extractor.dummy.DummyExtractor import DummyExtractor
from src.transformer.dummy.DummyTransformer import DummyTransformer

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        prog='ReflectDetect',
        description='Automatically detect reflection calibration panels in images and transform the given images to '
                    'reflectance',
        epilog='If you have any questions, please contact')
    parser.add_argument("path", help="Path to the image file or image files directory", type=str)
    args = parser.parse_args()

    detector = DummyDetector()
    extractor = DummyExtractor()
    transformer = DummyTransformer()


    def pipeline(path: str):
        logger.info(f"Transforming image {path} to reflectance")
        detection_results_path = detector.detect(path)
        extraction_results_path = extractor.extract(path, detection_results_path)
        transformed_image_path = transformer.transform(path, extraction_results_path)

        return transformed_image_path


    template_path = (Path.cwd() / args.path).resolve()
    if template_path.is_dir():
        for filename in template_path.glob("*"):
            pipeline(template_path.joinpath(filename))
    else:
        pipeline(template_path)
