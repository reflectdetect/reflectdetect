from abc import abstractmethod, ABC


class BaseExtractor(ABC):
    @abstractmethod
    def extract(self, image_path: str, detection_path: str) -> str:
        pass
