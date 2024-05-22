from src.extractor.BaseExtractor import BaseExtractor


class DummyExtractor(BaseExtractor):
    def get_name(self) -> str:
        return "dummy"

    def extract(self, image_path: str, detection_path: str) -> str:
        return "extractor/dummy/dummy_extractions.json"