from pydantic import BaseModel


class ApriltagPanelProperties(BaseModel):
    bands: list[float]
    tag_id: int
    family: str  # TODO: remove


class GeolocationPanelProperties(BaseModel):
    bands: list[float]
