from pydantic import BaseModel


class ApriltagPanelProperties(BaseModel):
    bands: list[float]
    tag_id: int
    family: str  # TODO: remove


class GeolocationPanelProperties(BaseModel):
    layer_name: str
    bands: list[float]
