import argparse
import logging
from pathlib import Path

from src.detector.BaseDetector import BaseDetector
from src.detector.dummy.DummyDetector import DummyDetector
from src.extractor.BaseExtractor import BaseExtractor
from src.extractor.dummy.DummyExtractor import DummyExtractor
from src.transformer.BaseTransformer import BaseTransformer
from src.transformer.dummy.DummyTransformer import DummyTransformer

logger = logging.getLogger(__name__)


def run_pipeline_for_each_image(detector: BaseDetector, extractor: BaseExtractor, transformer: BaseTransformer,
                                image_path: str):
    def pipeline(path: str):
        logger.info(f"Transforming image {path} to reflectance")
        detection_results_path = detector.detect(path)
        extraction_results_path = extractor.extract(path, detection_results_path)
        transformed_image_path = transformer.transform(path, extraction_results_path)

        return transformed_image_path, extraction_results_path, detection_results_path

    template_path = (Path.cwd() / image_path).resolve()
    file_endings = (".jpg", ".png", ".tif")
    if template_path.is_dir():
        images_path = (template_path / 'Images/seq1').resolve()
        results = []

        for e in file_endings:
            for filename in images_path.glob("*" + e):
                results.append((filename, *(pipeline(filename.absolute().as_posix()))))
        return results
    else:
        if template_path.name.endswith(file_endings):
            return list((template_path, pipeline(template_path.name)))
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

    detector = DummyDetector()
    extractor = DummyExtractor()
    transformer = DummyTransformer()

    run_pipeline_for_each_image(detector, extractor, transformer, args.path)
