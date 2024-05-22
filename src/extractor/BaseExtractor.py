from abc import abstractmethod, ABC


class BaseExtractor(ABC):
    @abstractmethod
    def extract(self, image_path: str, detection_path: str, panel_data_path: str) -> str:
        pass
