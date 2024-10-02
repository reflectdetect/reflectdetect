from unittest.mock import MagicMock, patch
from pathlib import Path
import pytest

from reflectdetect.utils.exif import get_camera_properties

@pytest.fixture
def mock_exiftool():
    """Fixture to mock ExifToolHelper."""
    mock_exiftool = MagicMock()
    return mock_exiftool

def test_get_camera_properties(mock_exiftool):
    """Test extraction of camera properties from EXIF data."""
    # Define a fake image path
    image_path = Path("/fake/path/image.jpg")

    # Mock the EXIF metadata that would be returned
    mock_metadata = [{
        "EXIF:FocalLength": 50.0,
        "EXIF:FocalPlaneXResolution": 3000.0,
        "EXIF:FocalPlaneYResolution": 2000.0,
        "EXIF:FocalPlaneResolutionUnit": 3
    }]

    # Mock the run_in_thread function to return the fake EXIF data
    with patch('reflectdetect.utils.exif.run_in_thread', return_value=mock_metadata):
        # Call the function with the mocked exiftool and image path
        focal_length, x_res, y_res, res_unit = get_camera_properties(mock_exiftool, image_path)

    # Assertions to check if the properties were correctly extracted
    assert focal_length == 50.0
    assert x_res == 3000.0
    assert y_res == 2000.0
    assert res_unit == 3
