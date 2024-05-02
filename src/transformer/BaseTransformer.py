from abc import ABC, abstractmethod


class BaseTransformer(ABC):

    @abstractmethod
    def transform(self, image_path, extraction_path) -> str:
        pass
