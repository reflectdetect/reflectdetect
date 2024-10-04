from pathlib import Path
from unittest import mock
from unittest.mock import patch, MagicMock

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from geopandas import GeoDataFrame
from rasterio._base import Affine
from rasterio.transform import from_origin
from shapely.geometry import Polygon

from reflectdetect.constants import ORTHOPHOTO_FOLDER
# Import the functions from your module
from reflectdetect.utils.orthophoto import (
    is_panel_in_orthophoto,
    extract_using_geolocation,
    save_bands,
    get_orthophoto_paths,
    load_panel_locations,
    build_batches_per_full_visibility,
    save_orthophotos,
)

# Constants used in the tests
NO_DATA_VALUE = -9999
SHRINK_FACTOR = 0.9


@pytest.fixture
def mock_rasterio_open(mocker):
    """Mock rasterio.open and its return object."""
    mock_dataset = mock.MagicMock()
    mock_dataset.__enter__.return_value = mock_dataset

    # Mock bounds and transform properties
    mock_dataset.bounds = (0, 0, 100, 100)  # Example bounds for the orthophoto
    mock_dataset.transform = from_origin(0, 100, 1, 1)  # Example transform (identity matrix)
    mock_dataset.width = 100
    mock_dataset.height = 100
    mock_dataset.meta = {
        'driver': 'GTiff',
        'dtype': 'float32',
        'nodata': None,
        'width': 100,
        'height': 100,
        'count': 3,
        'crs': '+proj=latlong',
        'transform': mock_dataset.transform
    }

    # Mock the context manager behavior
    mock_rasterio_open = mocker.patch("rasterio.open", return_value=mock_dataset)

    return mock_rasterio_open


@pytest.fixture
def mock_panel():
    """Create a mock panel as a GeoDataFrame with geometry."""
    # Create a simple square polygon within the bounds of the orthophoto
    panel_polygon = Polygon([(10, 10), (10, 20), (20, 20), (20, 10), (10, 10)])
    gdf = gpd.GeoDataFrame(geometry=[panel_polygon])
    return gdf


def test_is_panel_in_orthophoto_inside(mock_rasterio_open, mock_panel):
    """Test when the panel is inside the orthophoto bounds."""
    orthophoto_path = Path('/fake/path/orthophoto.tif')

    # Mock the output of the mask function
    mock_rasterio_open.read.return_value = np.array([[[0, 0], [0, 0]]], dtype=np.uint16)  # Dummy image data

    # Ensure rasterio.mask.mask is mocked to return expected values
    mock_mask = mock.MagicMock(return_value=(np.array([[[0, 0], [0, 0]]]), None))
    with pytest.MonkeyPatch.context() as m:
        m.setattr("rasterio.mask.mask", mock_mask)

        # Run the test
        result = is_panel_in_orthophoto(
            orthophoto_path, mock_panel, SHRINK_FACTOR, NO_DATA_VALUE
        )

    assert result is True


def test_is_panel_in_orthophoto_outside(mock_rasterio_open, mock_panel):
    """Test when the panel is outside the orthophoto bounds."""
    # Move the panel outside the raster bounds
    mock_panel.geometry = mock_panel.translate(200, 200).geometry
    orthophoto_path = Path('/fake/path/orthophoto.tif')
    result = is_panel_in_orthophoto(
        orthophoto_path, mock_panel, SHRINK_FACTOR, NO_DATA_VALUE
    )
    assert result is False

def test_get_orthophoto_paths(tmp_path):
    """Test retrieving orthophoto paths."""
    # Create fake orthophoto files
    orthophoto_folder = tmp_path / 'orthophotos'
    orthophoto_folder.mkdir()
    (orthophoto_folder / 'photo1.tif').touch()
    (orthophoto_folder / 'photo2.tif').touch()

    paths = get_orthophoto_paths(tmp_path, orthophoto_folder)
    assert len(paths) == 2
    assert all(path.suffix == '.tif' for path in paths)
    assert paths[0] == orthophoto_folder / 'photo1.tif'


def test_get_orthophoto_paths_default(tmp_path):
    """Test retrieving orthophoto paths with default folder."""
    # Create default orthophoto folder
    orthophoto_folder = tmp_path / ORTHOPHOTO_FOLDER
    orthophoto_folder.mkdir()
    (orthophoto_folder / 'photo1.tif').touch()
    (orthophoto_folder / 'photo2.tif').touch()

    paths = get_orthophoto_paths(tmp_path, None)
    assert len(paths) == 2
    assert paths[0] == orthophoto_folder / 'photo1.tif'


def test_load_panel_locations(tmp_path):
    """Test loading panel locations from a geopackage."""
    # Create a fake geopackage file with layers
    geopackage_path = tmp_path / 'panels.gpkg'

    # Mock fiona.listlayers and geopandas.read_file
    with mock.patch('fiona.listlayers') as mock_listlayers, \
            mock.patch('geopandas.read_file') as mock_read_file:
        mock_listlayers.return_value = ['layer1', 'layer2', 'android_metadata']
        mock_panel_gdf = gpd.GeoDataFrame({'geometry': [Polygon()]})
        mock_read_file.return_value = mock_panel_gdf

        panel_locations = load_panel_locations(tmp_path, geopackage_path)

        assert len(panel_locations) == 2  # Excludes 'android_metadata'
        assert all(isinstance(panel[1], GeoDataFrame) for panel in panel_locations)


def test_build_batches_per_full_visibility():
    """Test batching images based on panel visibility."""
    # Create a mock paths_with_visibility dictionary
    paths_with_visibility = {
        Path(f'image_{i}.tif'): np.array([i % 2 == 0, True, True])
        for i in range(5)
    }

    batches = build_batches_per_full_visibility(paths_with_visibility)
    assert isinstance(batches, list)
    for batch in batches:
        assert isinstance(batch, tuple)
        assert isinstance(batch[0], bool)
        assert isinstance(batch[1], list)


def test_save_orthophotos(tmp_path):
    """Test saving converted orthophotos."""
    dataset = tmp_path
    paths = [tmp_path / f'photo_{i}.tif' for i in range(3)]
    for path in paths:
        path.touch()

    converted_photos = [
        [np.random.rand(100, 100) for _ in range(3)],
        None,  # Simulate a photo that was skipped
        [np.random.rand(100, 100) for _ in range(3)],
    ]

    with mock.patch('rasterio.open') as mock_open, \
            mock.patch('reflectdetect.utils.orthophoto.save_bands') as mock_save_bands:
        mock_dataset = mock.MagicMock()

        mock_open.return_value.__enter__.return_value = mock_dataset

        save_orthophotos(dataset, paths, converted_photos)

        assert mock_open.call_count == 2  # Should skip one photo
        assert mock_save_bands.call_count == 2
