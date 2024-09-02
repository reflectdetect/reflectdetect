<br />
<p align="center">
<a><img src="https://raw.githubusercontent.com/reflectdetect/reflectdetect/main/images/logo-small.png" alt="ReflectDetect" width="128" height="128" title="ReflectDetect" style="border-radius: 50%;"></a>
</p>
<h3 align="center">ReflectDetect</h3>
<p align="center">    An automated workflow for in-flight radiometric calibration of UAV imagery<br /></p>
<p align="center">
  <a href="https://github.com/reflectdetect/reflectdetect/blob/main/LICENSE"><img src="https://img.shields.io/github/license/FawnRescue/frontend" alt="License"></a>
  <a href="https://github.com/reflectdetect/reflectdetect/network/members"><img src="https://img.shields.io/github/forks/reflectdetect/reflectdetect?style=social" alt="GitHub forks"></a>
  <a href="https://github.com/reflectdetect/reflectdetect/stargazers"><img src="https://img.shields.io/github/stars/reflectdetect/reflectdetect?style=social" alt="GitHub stars"></a>
</p>
<p>
<a href="https://github.com/reflectdetect/reflectdetect/issues">Report Bug</a>
·
<a href="https://github.com/reflectdetect/reflectdetect/issues">Request Feature</a>
</p>
<a href="https://reflectdetect.readthedocs.io">Documentation</a>

## Overview

Welcome to the official repository for the paper, "Application Note: An automated workflow for in-flight radiometric
calibration of UAV imagery".

<!-- TODO: Properly cite paper -->

### Abstract

> UAVs equipped with optical sensors have transformed remote sensing in vegetation science
> by providing high-resolution, on-demand data, enhancing studies in forestry,
> agriculture, and environmental monitoring.
> However, accurate radiometric calibration of UAV imagery remains challenging due to environmental variability
> and the limitations of existing methods, such as the common single gray reference panel, which is prone to errors.
> ReflectDetect, an open-source tool, addresses these challenges
> by automating radiometric calibration using geotagging and AprilTag detection.
> This dual-module approach ensures reliable calibration under varying conditions, reduces human error, and increases
> efficiency through a user-friendly CLI.
> ReflectDetect's modular design supports future enhancements and broader applicability across research fields.

<!-- TODO: Add images -->

> [!NOTE]
> We provide two workflows. For a detailed look at the technical details make
> sure to follow the upcoming `setup` and `usage` sections for your preferred workflow.

### :artificial_satellite: Workflow 1: Geolocation-Based Calibration

1. **Panel Reflectance Data**: Gather reflectance values of the panels for the bands you will capture images in, either
   from the manufacturer (for commercial panels) or using a field spectrometer (for DIY panels). Save these values in
   a `panel_properties.json` file.
2. **Field Setup**: Position the calibration panels in the field.
3. **Panel Location Data**: Capture the exact locations of the panel corners (we used a [Device Name](#) in our testing)
   and save them in a `panel_locations.gpkg` file.
4. **Image Capture**: Fly your drone mission, capturing images.
5. **Image Processing**: Convert the captured images to orthophotos by rectifying and geo-referencing them (we
   used [Software Name](#) in our testing).
6. **Run ReflectDetect**: Use ReflectDetect to convert the orthophotos to reflectance data.

### :white_square_button: Workflow 2: AprilTag-Based Calibration

1. **Panel Reflectance Data**: Gather reflectance values of the panels for the bands you will capture images in, either
   from the manufacturer (for commercial panels) or using a field spectrometer (for DIY panels). Save these values in
   a `panel_properties.json` file.
2. **Print AprilTags**: Print an AprilTag for each panel. PDF files for printing are available in
   the `/apriltag_printouts/` directory of this repository.
   <!-- [TODO: Add PDF files] -->
3. **Field Setup**: Position the calibration panels and place the AprilTags according to the placement guide.
4. **Image Capture**: Fly your drone mission, capturing images.
5. **Run ReflectDetect**: Use ReflectDetect to convert the images to reflectance data.

### Key Concepts and Vocabulary

- **Panel**: Calibration sheets placed in the field, used to compare captured intensity values in images with known
  reflectance values in the `panel_properties.json` file.

- **AprilTag / Tag**: A visual marker used for accurate detection in images.

- **Field**: The area captured by drone imagery.

- **Image**: Individual bands of captured data. ReflectDetect assumes all images were taken at equal time intervals and
  named in the format `*_{band_index}.tif`.  
  For example, `IMG_0052_6.tif` indicates the 6th band.

- **Orthophoto / Photo**: An image where all bands are combined and geo-referenced.

## Installation

### Installing Python

To get started with this project, you'll need to have Python installed on your system. Follow the steps below to install
Python:

#### 1. Check if Python is Already Installed

Before installing, check if Python is already installed on your system:

- **Windows/Mac/Linux:**

  Open a terminal or command prompt and run the following command:

  ```sh
  python --version
  ```

  or

  ```sh
  python3 --version
  ```

  If Python is installed, you will see a version number. (Python 3.10 or higher is required)

#### 2. Download and Install Python

If Python is not installed, follow these steps:

- **Windows:**

    1. Go to the official [Python website](https://www.python.org/downloads/).
    2. Download the latest version for Windows.
    3. Run the installer. Make sure to check the option "Add Python to PATH" during installation.
    4. Complete the installation process.

- **Mac:**

    1. Download the latest version of Python from the [Python website](https://www.python.org/downloads/).
    2. Open the downloaded `.pkg` file and follow the instructions to install.
    3. Alternatively, you can use Homebrew:
       ```sh
       brew install python
       ```

- **Linux:**

    1. Use the package manager for your distribution (e.g., `apt` for Ubuntu):
       ```sh
       sudo apt update
       sudo apt install python3
       ```
    2. For other distributions, refer to your package manager's documentation.

#### 3. Verify Installation

After installation, verify that Python was installed correctly:

- Open a terminal or command prompt and run:

  ```sh
  python --version
  ```

  or

  ```sh
  python3 --version
  ```

  You should see the version of Python that you installed.
  Now you're ready to install project dependencies and start coding!

### Installing ExifTool

ExifTool is an essential tool for working with image metadata in ReflectDetect. Follow the instructions below to install
ExifTool on your system.

#### Windows Installation

1. **Download ExifTool**:
    - Visit the [ExifTool Home Page](https://exiftool.org) and download the latest 64bit Windows
      Executable (`exiftool-xx.xx_64.zip`).

2. **Extract the Zip File**:
    - Double-click the downloaded `.zip` file to open it, then drag the `exiftool-xx.xx_64x` folder to your Desktop (
      where `xx` represents the version).

3. **Prepare for Command Line Use**:
    - Open the `exiftool-12.96_xx` folder on your Desktop.
    - Rename the `exiftool(-k).exe` file to `exiftool.exe`.
    - Move the folder folder to a convenient location (e.g., `C:\ExifTool\` or another folder of your choice).

4. **Add the Folder to the PATH**:
    - Open the Start Menu, search for "Environment Variables" and select Edit the system environment variables.
    - In the window that opens, click the Environment Variables button.
    - Under the User variables section, find the variable named Path and select it. Then click Edit.
    - In the new window, click New and enter the path to the folder where you moved exiftool.exe (e.g., C:\ExifTool).
    - Click OK to close all the windows.

5. **Verify Installation**:
    - Open Command Prompt (`cmd`) and type `exiftool`. You should see ExifTool's help information, confirming it is
      installed and recognized by your system.

#### macOS Installation

1. **Download ExifTool**:
    - Visit the [ExifTool Home Page](https://exiftool.org) and download the latest MacOS Package (`ExifTool-12.96.pkg`).

2. **Install ExifTool**:
    - Double-click the downloaded `.pkg` file and follow the installation instructions.

3. **Verify Installation**:
    - Open Terminal and type `exiftool`. You should see ExifTool's help information, confirming it is installed and
      recognized by your system.

#### Linux Installation

1. **Install via Package Manager**:
    - On most Linux distributions, you can install ExifTool using the package manager. For example, on Ubuntu, run:
      ```sh
      sudo apt-get install exiftool
      ```

2. **Verify Installation**:
    - Open your terminal and type `exiftool`. You should see ExifTool's help information, confirming it is installed and
      recognized by your system.

### Installing reflectdetect

```
pip install reflectdetect
```

# :white_square_button: Apriltags

## :white_square_button: Setup

### Create a panel_properties.json file

To give access to the information about your calibration panels, create a `panel_properties.json` file. It
includes the reflectance values of each panel for each of the bands you captured.
The following example shows how two panels might be configured. All the information about the first panel is
between the first `{ }` and so on.

The tag id has to correspond to the id of the apriltag you placed next to the given panel. No id can be used twice!

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
  }
]
```

### Create dataset folder

In order for reflect-detect to be able to gather the necessary information about the images, panels, camera, etc. ,
reflect-detect expects you to structure your data in the following format:

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
> If any of the folders/files are located elsewhere, you can specify their location using the `--panel_properties_file`
> or `--images_folder` argument

## :white_square_button: Usage

```bash
python .\reflectdetect\apriltag_main.py --family "tag25h9" --panel_properties_file data/apriltags_run2/panel_properties.json --images_folder data/apriltags_run2/0001SET/000 ".\data\apriltags_run2\0001SET\000\" -d
```

# :artificial_satellite: Geolocation

## :artificial_satellite: Setup

### Create a panel_properties.json file

To access the information about your calibration panels, we need you to create a `panel_properties.json` file. It
includes the reflectance values of each panel for each of the bands you captured.
In the following example we show how two panels might be configured. All the information about the first panel is
between the first `{ }` and so on.

We assume that the first panel in the file corresponds to the first layer of coordinates in the geopackage file (TODO:
explain better)

```json filename="panel_properties.json"
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
  }
]
````

### Create dataset folder

In order for reflect-detect to be able to gather the necessary information about the images, panels, camera, etc. ,
reflect-detect expects you to structure your data in the following format:

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

# Planned Features

- [] Support for unequal time intervals between images
- [] Customize parameters on a per panel basis

# Contributing

# Gallery

  <!--TODO: Add a good image of the panels]-->
  <!--TODO: Add a good image of the apriltags]-->
  <!--TODO: Add a good image of the captured images]-->

# AI Usage Card

# References

# License