from pathlib import Path
from typing import Any
from unittest import mock
from unittest.mock import MagicMock

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

# Assuming the functions are imported from the reflectdetect module
from reflectdetect.utils.apriltags import (
    verify_detections,
    build_batches_per_band,
    get_panel,
    save_images,
)


# Mocking AprilTagDetection and related objects
# noinspection PyUnusedLocal
class MockAprilTagDetection:
    def __init__(self, tag_id: Any, corners: Any, center: Any, family: str = "tag36h11") -> None:
        self.tag_id = tag_id
        self.corners = corners
        self.center = center
        self.family: str = family

    def getId(self) -> Any:
        return self.tag_id

    def getCorners(self, default: Any) -> Any:
        return self.corners

    def getCenter(self) -> Any:
        class Center(object):
            x = self.center[0]
            y = self.center[1]

        return Center()

    def getFamily(self) -> str:
        return self.family


def test_verify_detections() -> None:
    """Test that only valid tag ids are accepted."""
    tag = MockAprilTagDetection(1, [], (0, 0))  # type: ignore
    valid_ids = [1, 2, 3]
    # noinspection PyTypeChecker
    assert verify_detections(tag, valid_ids) is True  # type: ignore

    tag = MockAprilTagDetection(4, [], (0, 0))  # type: ignore
    # noinspection PyTypeChecker
    assert verify_detections(tag, valid_ids) is False  # type: ignore


def test_build_batches_per_band() -> None:
    """Test if batches are built correctly based on file naming."""
    paths = [
        Path("image_1_radiance.tif"),
        Path("image_2_radiance.tif"),
        Path("image_1234_2_radiance.tif"),
    ]
    batches = build_batches_per_band(paths)
    assert len(batches) == 2
    assert len(batches[0]) == 1
    assert len(batches[1]) == 2

    with pytest.raises(Exception):
        build_batches_per_band([Path("image_radiance.tif")])


@given(
    tag_smudge_factor=st.floats(min_value=0.1, max_value=2.0),
    panel_size_pixel=st.tuples(st.integers(10, 50), st.integers(10, 50)),
)
def test_get_panel(tag_smudge_factor: float, panel_size_pixel: tuple[int, int]) -> None:
    """Test getting panel corners from a tag detection."""
    tag = MockAprilTagDetection(1, [0, 0, 10, 0, 10, 10, 0, 10], (5, 5))  # type: ignore
    # noinspection PyTypeChecker
    corners = get_panel(
        tag,  # type: ignore
        panel_size_pixel=panel_size_pixel,
        image_dimensions=(100, 100),
        tag_smudge_factor=tag_smudge_factor,
        tag_direction="up",
    )

    # The tags lower left corner is in the corner of the whole image ((0,0)).
    # Therefore panels with a width larger than 10
    # are clipping the left border of the image and should be discarded.
    is_in_bounds = panel_size_pixel[0] <= 10
    if is_in_bounds:
        assert corners is not None  # Should return corners within bounds
    else:
        assert corners is None


@mock.patch("reflectdetect.utils.apriltags.imwrite")
def test_save_images(mock_imwrite: MagicMock, tmp_path: Path) -> None:
    """Test saving images with a mock EXIF tool."""
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    paths = [dataset / "image1.tif", dataset / "image2.tif", dataset / "image3.tif"]
    converted_images = [np.random.rand(100, 100), np.random.rand(100, 100), None]
    exiftool = mock.MagicMock()

    save_images(exiftool, dataset, paths, converted_images)

    assert exiftool.execute.called
    assert mock_imwrite.called
    assert mock_imwrite.call_count == 2
