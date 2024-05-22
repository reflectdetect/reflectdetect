from abc import abstractmethod, ABC
from typing import Iterable


class BaseBatchExtractor(ABC):

    @abstractmethod
    def get_name(self) -> str:
        return "base_batch"

    @abstractmethod
    def extract(self, image_paths: Iterable, detection_path: str, panel_data_path: str) -> str:
        pass
