from abc import abstractmethod, ABC
from typing import Iterable


class BaseBatchDetector(ABC):
    @abstractmethod
    def detect(self, image_paths: Iterable) -> str:
        pass
