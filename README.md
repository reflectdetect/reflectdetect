# reflectdetect

Automatic detection of reflectance calibration panels in multiband drone imagery
TODO Explain both approaches and add images

We provide two workflows
1. :artificial_satellite: Geolocation
2. :white_square_button: Apriltags


## Vocabulary
### Panel
With the word panel we reference calibration sheets TODO
### Tag

## Installation
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
# :artificial_satellite: Geolocation
## :artificial_satellite: Setup
### :artificial_satellite: Create a panel_properties.json file
To access the information about your calibration panels, we need you to create a `panel_properties.json` file. It includes the reflectance values of each panel for each of the bands you captured.
In the following example we show how two panels might be configured. All the information about the first panel is between the first `{ }` and so on.

We assume that the first panel in the file corresponds to the first layer of coordinates in the geopackage file (TODO: explain better)

```json
[
  {
    "layer_id": 0,
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
    "layer_id": 1,
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
│   panel_locations.gpk
│
└───orthophotos
│   │   IMG_0000.tif
│   │   IMG_0001.tif
│   │   IMG_0002.tif
│   │   IMG_0003.tif
|   |   ...
```

