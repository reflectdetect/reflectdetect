from typing import Any

from pydantic import BaseModel
from rich.console import Console
from rich.table import Table

from reflectdetect.utils.paths import default


class ApriltagPanelProperties(BaseModel):
    bands: list[float]
    tag_id: int
    panel_width: float | None = None
    panel_height: float | None = None
    tag_smudge_factor: float | None = None
    panel_smudge_factor: float | None = None
    tag_family: str | None = None
    tag_direction: str | None = None
    shrink_factor: float | None = None


class ValidatedApriltagPanelProperties(BaseModel):
    bands: list[float]
    tag_id: int
    panel_width: float
    panel_height: float
    tag_smudge_factor: float
    panel_smudge_factor: float
    tag_family: str
    tag_direction: str
    shrink_factor: float


class ApriltagPanelPropertiesFile(BaseModel):
    panel_properties: list[ApriltagPanelProperties]
    tag_size: float
    default_panel_width: float | None = None
    default_panel_height: float | None = None
    default_tag_family: str | None = None
    default_tag_direction: str | None = None
    default_panel_smudge_factor: float | None = None
    default_tag_smudge_factor: float | None = None
    default_shrink_factor: float | None = None
    exclude: list[str] | None = None


class GeolocationPanelProperties(BaseModel):
    bands: list[float]
    layer_name: str
    panel_width: float | None = None
    panel_height: float | None = None
    panel_smudge_factor: float | None = None
    shrink_factor: float | None = None


class ValidatedGeolocationPanelProperties(BaseModel):
    bands: list[float]
    layer_name: str
    panel_width: float
    panel_height: float
    panel_smudge_factor: float
    shrink_factor: float


class GeolocationPanelPropertiesFile(BaseModel):
    panel_properties: list[GeolocationPanelProperties]
    default_panel_width: float | None = None
    default_panel_height: float | None = None
    default_panel_smudge_factor: float | None = None
    default_shrink_factor: float | None = None


def validate_apriltag_panel_properties(
        panels: list[ApriltagPanelProperties], default_properties: dict[str, Any]
) -> list[ValidatedApriltagPanelProperties]:
    validated_panel_properties: list[ValidatedApriltagPanelProperties] = []

    for index, panel in enumerate(panels):

        def set_with_default(value: Any | None, name: str) -> Any:
            result = (
                value if value is not None else default_properties["default_" + name]
            )
            if result is None:
                raise Exception(
                    f"Panel {index + 1}: {name} not set and no default supplied"
                )
            return result

        bands = panel.bands

        panel_width: float = set_with_default(panel.panel_width, "panel_width")
        if panel_width <= 0.0:
            raise Exception(f"Panel {index + 1}: Panel width must be greater than zero")
        panel_height: float = set_with_default(panel.panel_height, "panel_height")
        if panel_height <= 0.0:
            raise Exception(
                f"Panel {index + 1}: Panel height must be greater than zero"
            )
        tag_id: int = panel.tag_id
        if tag_id < 0:
            raise Exception(f"Panel {index + 1}: tag_id can not be negative")
        tag_family: str = set_with_default(panel.tag_family, "tag_family")
        tag_smudge_factor: float = set_with_default(
            panel.tag_smudge_factor, "tag_smudge_factor"
        )
        panel_smudge_factor: float = set_with_default(
            panel.panel_smudge_factor, "panel_smudge_factor"
        )
        tag_direction: str = set_with_default(panel.tag_direction, "tag_direction")
        shrink_factor: float = set_with_default(panel.shrink_factor, "shrink_factor")
        if shrink_factor <= 0.0:
            raise Exception(
                f"Panel {index + 1}: shrink_factor must be greater than zero"
            )
        if shrink_factor > 1.0:
            raise Exception(f"Panel {index + 1}: shrink_factor must be smaller than 1")
        validated_panel_properties.append(
            ValidatedApriltagPanelProperties(
                bands=bands,
                panel_width=panel_width,
                panel_height=panel_height,
                tag_id=tag_id,
                tag_family=tag_family,
                tag_smudge_factor=tag_smudge_factor,
                panel_smudge_factor=panel_smudge_factor,
                tag_direction=tag_direction,
                shrink_factor=shrink_factor
            )
        )
    print_panel_properties(validated_panel_properties)
    return validated_panel_properties


def validate_geolocation_panel_properties(
        panels: list[GeolocationPanelProperties], default_properties: dict[str, Any]
) -> list[ValidatedGeolocationPanelProperties]:
    validated_panel_properties: list[ValidatedGeolocationPanelProperties] = []

    for index, panel in enumerate(panels):

        def set_with_default(value: Any | None, name: str) -> Any:
            result = (
                value if value is not None else default_properties["default_" + name]
            )
            if result is None:
                raise Exception(
                    f"Panel {index + 1}: {name} not set and no default supplied"
                )
            return result

        bands = panel.bands

        panel_width: float = set_with_default(panel.panel_width, "panel_width")
        if panel_width <= 0.0:
            raise Exception(f"Panel {index + 1}: Panel width must be greater than zero")
        panel_height: float = set_with_default(panel.panel_height, "panel_height")
        if panel_height <= 0.0:
            raise Exception(
                f"Panel {index + 1}: Panel height must be greater than zero"
            )
        panel_smudge_factor: float = set_with_default(
            panel.panel_smudge_factor, "panel_smudge_factor"
        )
        shrink_factor: float = set_with_default(panel.shrink_factor, "shrink_factor")
        if shrink_factor <= 0.0:
            raise Exception(
                f"Panel {index + 1}: shrink_factor must be greater than zero"
            )
        if shrink_factor > 1.0:
            raise Exception(f"Panel {index + 1}: shrink_factor must be smaller than 1")
        layer_name = panel.layer_name
        validated_panel_properties.append(
            ValidatedGeolocationPanelProperties(
                bands=bands,
                panel_width=panel_width,
                panel_height=panel_height,
                panel_smudge_factor=panel_smudge_factor,
                shrink_factor=shrink_factor,
                layer_name=layer_name,
            )
        )
    print_panel_properties(validated_panel_properties)
    return validated_panel_properties


def print_panel_properties(
        panel_properties: list[ValidatedApriltagPanelProperties] | list[ValidatedGeolocationPanelProperties]) -> None:
    table = Table(title="Computed Panel Properties (File > CLI Arguments > Default Value)")
    properties = [prop.model_dump(exclude=set("bands")) for prop in panel_properties]
    for key in properties[0].keys():
        table.add_column(key.capitalize(), justify="left", style="cyan", no_wrap=True)

    # Add rows dynamically based on the values
    for item in properties:
        row = [str(item[key]) for key in item.keys()]  # Convert all values to strings
        table.add_row(*row)

    # Print the table
    console = Console()
    console.print(table)
