from abc import ABC, abstractmethod


class BaseTransformer(ABC):

    @abstractmethod
    def get_name(self) -> str:
        return "base"

    @abstractmethod
    def transform(self, image_path, extraction_path) -> str:
        pass
