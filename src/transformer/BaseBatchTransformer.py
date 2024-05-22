from abc import ABC, abstractmethod
from typing import Iterable


class BaseBatchTransformer(ABC):

    @abstractmethod
    def transform(self, image_paths: Iterable, extraction_path) -> str:
        pass
