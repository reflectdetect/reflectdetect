import argparse
import logging
from itertools import groupby
from pathlib import Path

from src.detector.BaseBatchDetector import BaseBatchDetector
from src.detector.BaseDetector import BaseDetector
from src.detector.dummy.DummyDetector import DummyDetector
from src.extractor.BaseExtractor import BaseExtractor
from src.extractor.dummy.DummyExtractor import DummyExtractor
from src.transformer.BaseTransformer import BaseTransformer
from src.transformer.dummy.DummyTransformer import DummyTransformer
from src.utils.paths import get_image_id

logger = logging.getLogger(__name__)


def run_pipeline_for_each_image(detector: BaseDetector, extractor: BaseExtractor, transformer: BaseTransformer,
                                image_path: str):
    def pipeline(path: str):
        logger.info(f"Transforming image {path} to reflectance")
        detection_results_path = detector.detect(path)
        extraction_results_path = extractor.extract(path, detection_results_path)
        transformed_image_path = transformer.transform(path, extraction_results_path)

        return transformed_image_path

    template_path = (Path.cwd() / image_path).resolve()
    if template_path.is_dir():
        for filename in template_path.glob("*"):
            pipeline(template_path.joinpath(filename).name)
    else:
        pipeline(template_path.name)


def run_pipeline_for_batch(detector: BaseBatchDetector,
                           extractor: BaseExtractor,
                           transformer: BaseTransformer,
                           image_path: str):
    def pipeline(image_paths: [str]):
        logger.info(f"Transforming images {get_image_id(image_paths[0])}_* to reflectance")
        detection_results_path = detector.detect(image_paths)
        extraction_results_path = extractor.extract(image_paths, detection_results_path)
        transformed_image_path = transformer.transform(image_paths, extraction_results_path)

        return transformed_image_path

    # create batches
    batches = []
    template_path = (Path.cwd() / image_path).resolve()
    if template_path.is_dir():
        # TODO: filter for only images
        # group image paths by image id
        grouped_images = [list(v) for i, v in groupby(template_path.glob("*"), lambda x: get_image_id(x))]
        batches.append(*grouped_images)
    else:
        batches.append([template_path])

    for batch in batches:
        pipeline(batch)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        prog='ReflectDetect',
        description='Automatically detect reflection calibration panels in images and transform the given images to '
                    'reflectance',
        epilog='If you have any questions, please contact')
    parser.add_argument("path", help="Path to the image file or image files directory", type=str)
    args = parser.parse_args()

    d = DummyDetector()
    e = DummyExtractor()
    t = DummyTransformer()

    run_pipeline_for_each_image(d, e, t, args.path)
