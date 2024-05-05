from abc import abstractmethod, ABC


class BaseDetector(ABC):
    @abstractmethod
    def detect(self, image_path: str) -> str:
        pass
