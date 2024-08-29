# reflectdetect

Automatic detection of reflectance calibration panels in multiband drone imagery
## Vocabulary
### Panel
With the word panel we reference calibration sheets TODO
### Tag

## Installation
## Setup
### Create dataset folder
In order for reflect-detect to be able to gather the necessary information about the images, panels, camera, etc. , reflect-detect expects you to structure your data in the following format:
#### Geolocation
```
dataset_folder
│   panels_properties.json
│   panel_locations.gpk
│
└───images
│   │   IMG_0000_1.tif
│   │   IMG_0000_2.tif
│   │   IMG_0001_1.tif
│   │   IMG_0001_2.tif
|   |   ...
│   
└───ortho
    │   file021.txt
    │   file022.txt 
```

either images or orthophotos
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
   - 

