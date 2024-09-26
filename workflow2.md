# Workflow 2: AprilTag-Based Calibration

# Apriltag Printing

# Apriltag Placement Guide

# Apriltag Measurement Guide

See: `apriltag_area_measurement.ipynb`

# Apriltag Setup

## Create dataset folder

In order for reflect-detect to be able to gather the necessary information about the images, panels, etc. ,
reflect-detect expects you to structure your data in the following format:

```
dataset_folder
│   panels_properties.json
│   
└───raw
│   │   IMG_0000_1.tif
│   │   IMG_0000_2.tif
│   │   IMG_0001_1.tif
│   │   IMG_0001_2.tif
|   |   ...
```

## Convert raw images to radiance

Use the `reflectdetect-converter` CLI script to convert the raw images into radiance.
Move into you dataset folder

```bash
cd C:\Users\username\Desktop\dataset_folder
```

Run the converted while specifying the manufacturer of your camera. Check supported manufacturers
using `reflectdetect-converter -h`.
If your manufacturer is not supported choose `generic`

```bash
reflectdetect-converter --manufacturer generic
```

The radiance images should now be ready for conversion in the `images` subfolder.

## Create a panel_properties.json file

To give access to the information about your calibration panels, create a `panel_properties.json` file. It
includes the reflectance values of each panel for each of the bands you captured.
The following examples show how two panels might be configured.

The first panel will use the default values, while the second panel specifies some of its own parameters.
Only `default_panel_width` and `default_panel_height` are required, the other parameters will have the default values as
below if not specified.
The tag id has to correspond to the id of the apriltag you placed next to the given panel. No id can be used twice!

```json
{
  "default_panel_width": 1.3,
  "default_panel_height": 1.3,
  "default_tag_family": "tag25h9",
  "default_tag_direction": "up",
  "default_panel_smudge_factor": 0.8,
  "default_tag_smudge_factor": 1.0,
  "default_shrink_factor": 0.8,
  "panel_properties": [
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
      "panel_width": 2.0,
      "shrink_factor": 0.5,
      "bands": [
        0.9,
        1.0,
        0.43,
        0.70,
        0.35,
        0.4
      ]
    }
  ]
}
```

> [!TIP]
> The default parameters can also be set using CLI arguments if not specified in the properties file
>
> Run ```reflectdetect-apriltag -h``` to get a list

# Apriltag Usage

After preparing the dataset folder, you are ready to run reflectdetect.
Open a command line or terminal.

To print the available arguments, run `reflectdetect-apriltag --help`

Assuming:

- the prepared dataset folder is at `C:\Users\username\Desktop\dataset_folder`
- the `panel_properties.json` file and the `images` folder are in the dataset folder and correctly named

## Minimal example:

To start the program, open your terminal or command line and run

```bash
cd C:\Users\username\Desktop\dataset_folder
```

then

```bash
reflectdetect-apriltag
```