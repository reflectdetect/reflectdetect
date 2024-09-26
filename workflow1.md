# Workflow 1: Geolocation-Based Calibration

# Geolocation Setup

## Create dataset folder

To easily run reflectdetect, structure your data in the following format:

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

## Create a panel_properties.json file

To access the information about your calibration panels, we need you to create a `panel_properties.json` file. It
includes the reflectance values of each panel for each of the bands you captured.
In the following example we show how two panels might be configured. All the information about the first panel is
between the first `{ }` and so on.

The layer name corresponds to the name of the layer the coordinates of the panel corners are stored in, in the
`panel_locations.gpkg` file

```json
{
  "default_panel_width": 1.3,
  "default_panel_height": 1.3,
  "panel_properties": [
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
    }
  ]
}
````

# Geolocation Usage

After preparing the dataset folder, you are ready to run reflectdetect.
Open a command line or terminal.

To print the available arguments, run `reflectdetect-geolocation --help`

Assuming:

- the prepared dataset folder is at `C:\Users\username\Desktop\dataset_folder`
- the `panel_properties.json` and `panel_locations.gpkg` files and the `images` folder are in the dataset folder and
  correctly named

## Minimal example:

To start the program, open your terminal or command line and run

```bash
cd C:\Users\username\Desktop\dataset_folder
```

then

```bash
reflectdetect-geolocation
```

Alternatively you can run the program from anywhere using

```bash
reflectdetect-geolocation "C:\Users\username\Desktop\dataset_folder"
```