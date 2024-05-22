from src.transformer.BaseTransformer import BaseTransformer


class DummyTransformer(BaseTransformer):
    def get_name(self) -> str:
        return "dummy"

    def transform(self, image_path, extraction_path) -> str:
        return image_path
