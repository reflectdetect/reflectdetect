# reflectdetect

Automatic detection of reflectance calibration panels in multiband drone imagery
TODO Explain both approaches and add images

We provide two workflows
### 1. :artificial_satellite: Geolocation
   - place panels in field
   - capture location of panel corners and save them to .gpkg file
   - fly drone and capture images 
   - convert images to orthophotos by rectifying and georeferencing them
   - run reflect detect on photos to convert them to reflectance images
### 2. :white_square_button: Apriltags
   - print out apriltags
   - place them according to the placement guide
   - fly drone and capture images 
   - run reflect detect on images to convert them to reflectance images


## Vocabulary
### Panel
With the word panel we reference calibration sheets
### Tag
Apriltag

#### Primary detection area vs total area
### Field
the area the drone images capture
### Image
### Orthophoto / Photo

## Installation
TODO: Install python
```
pip install reflectdetect
```

# :white_square_button: Apriltags
## :white_square_button: Setup
### :white_square_button: Create a panel_properties.json file
To access the information about your calibration panels, we need you to create a `panel_properties.json` file. It includes the reflectance values of each panel for each of the bands you captured.
In the following example we show how two panels might be configured. All the information about the first panel is between the first `{ }` and so on.

The tag id has to correspond to the id of the apriltag you place next to the given panel. No id can be used twice!
```json
[
  {
    "tag_id": 1,
    "bands": [
      0.9,
      1.0,
      1.0,
      1.0,
      0.72,
      0.91
    ]
  },
  {
    "tag_id": 2,
    "bands": [
      0.9,
      1.0,
      0.43,
      0.70,
      0.35,
      0.4
    ]
  },
  ...
]
````
### :white_square_button: Create dataset folder
In order for reflect-detect to be able to gather the necessary information about the images, panels, camera, etc. , reflect-detect expects you to structure your data in the following format:
```
dataset_folder
│   panels_properties.json
│   
└───images
│   │   IMG_0000_1.tif
│   │   IMG_0000_2.tif
│   │   IMG_0001_1.tif
│   │   IMG_0001_2.tif
|   |   ...
```

> [!TIP]
> If any of the folders/files are located elsewhere, you can specify their location using the `--panel_properties_file` or `--images_folder` argument

__either images or orthophotos
images are one band at a time
images are assumed to be named "*_{band_index}.tif"
we use the regec
so for examalpe IMG_0052_6.tif -> 6
regexr.com/857r5
orthophotos are all bands in one
panel properties file
include the reflectance values for each band
   - images
   - panel_properties
   -__ 
## :white_square_button: Usage
```bash
python .\reflectdetect\apriltag_main.py --family "tag25h9" --panel_properties_file data/apriltags_run2/panel_properties.json --images_folder data/apriltags_run2/0001SET/000 ".\data\apriltags_run2\0001SET\000\" -d
```
# :artificial_satellite: Geolocation
## :artificial_satellite: Setup
### :artificial_satellite: Create a panel_properties.json file
To access the information about your calibration panels, we need you to create a `panel_properties.json` file. It includes the reflectance values of each panel for each of the bands you captured.
In the following example we show how two panels might be configured. All the information about the first panel is between the first `{ }` and so on.

We assume that the first panel in the file corresponds to the first layer of coordinates in the geopackage file (TODO: explain better)

```json
[
  {
    "layer_name": "corner_27",
    "bands": [
      0.9,
      1.0,
      1.0,
      1.0,
      0.72,
      0.91
    ]
  },
  {
    "layer_name": "corner_28",
    "bands": [
      0.9,
      1.0,
      0.43,
      0.70,
      0.35,
      0.4
    ]
  },
  ...
]
````
### :artificial_satellite: Create dataset folder
In order for reflect-detect to be able to gather the necessary information about the images, panels, camera, etc. , reflect-detect expects you to structure your data in the following format:
```
dataset_folder
│   panels_properties.json
│   panel_locations.gpkg
│
└───orthophotos
│   │   IMG_0000.tif
│   │   IMG_0001.tif
│   │   IMG_0002.tif
│   │   IMG_0003.tif
|   |   ...
```

## :artificial_satellite: Usage