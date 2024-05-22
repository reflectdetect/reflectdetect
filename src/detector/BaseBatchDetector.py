from abc import abstractmethod, ABC
from typing import Iterable


class BaseBatchDetector(ABC):

    @abstractmethod
    def get_name(self) -> str:
        return "base_batch"

    @abstractmethod
    def detect(self, image_paths: Iterable) -> str:
        pass
