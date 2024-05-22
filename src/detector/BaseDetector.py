from abc import abstractmethod, ABC


class BaseDetector(ABC):

    @abstractmethod
    def get_name(self) -> str:
        return "base"

    @abstractmethod
    def detect(self, image_path: str) -> str:
        pass
