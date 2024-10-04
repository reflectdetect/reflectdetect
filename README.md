<br />
<p align="center">
<a><img src="https://raw.githubusercontent.com/reflectdetect/reflectdetect/main/images/logo-small.png" alt="reflectdetect" width="128" height="128" title="reflectdetect" style="border-radius: 50%;"></a>
</p>
<h3 align="center">reflectdetect</h3>
<p align="center">    An automated workflow for in-flight radiometric calibration of UAV imagery<br /></p>
<p align="center">
    <img alt="Coverage" src="https://img.shields.io/badge/Coverage-30%25-brightgreen">
  <a href="https://github.com/reflectdetect/reflectdetect/blob/main/LICENSE"><img src="https://img.shields.io/github/license/reflectdetect/reflectdetect" alt="License"></a>
  <a href="https://github.com/reflectdetect/reflectdetect/network/members"><img src="https://img.shields.io/github/forks/reflectdetect/reflectdetect?style=social" alt="GitHub forks"></a>
  <a href="https://github.com/reflectdetect/reflectdetect/stargazers"><img src="https://img.shields.io/github/stars/reflectdetect/reflectdetect?style=social" alt="GitHub stars"></a>
</p>
<p>
<a href="https://github.com/reflectdetect/reflectdetect/issues">Report Bug</a>
·
<a href="https://github.com/reflectdetect/reflectdetect/issues">Request Feature</a>
</p>
<a href="https://reflectdetect.readthedocs.io">Documentation</a>
</p>
<a href="https://ifzgoettingen-my.sharepoint.com/:f:/g/personal/heim_ifz-goettingen_de/EmSDi-poSitJpDbd4Xrrhj8BF-9x5LMSKSMOvJbj5OqmXg?e=g0oss6">Example Data</a>

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
> reflectdetect, an open-source tool, addresses these challenges
> by automating radiometric calibration using geotagging and AprilTag detection.
> This dual-module approach ensures reliable calibration under varying conditions, reduces human error, and increases
> efficiency through a user-friendly CLI.
> reflectdetect's modular design supports future enhancements and broader applicability across research fields.

### Workflow 1: Geolocation-Based Calibration

1. **Gather Panel Reflectance Data**  
   For the bands you will capture images in, either
   from the manufacturer (for commercial panels) or using a field spectrometer (for DIY panels). Save these values in
   a `panel_properties.json` file.
2. **Position the calibration panels in the field**
3. **Capture the locations of the panel corners**  
   We used a [Device Name](#) in our testing, save the data in a `panel_locations.gpkg` file.
4. **Fly your drone mission, capturing images.**
5. **Rectifying and geo-reference the images**
   <br>We used [Software Name](#) in our testing
6. **Run reflectdetect**:
   <br>Convert the orthophotos to reflectance data.

### Workflow 2: AprilTag-Based Calibration

1. **Gather Panel Reflectance Data**  
   For the bands you will capture images in, either
   from the manufacturer (for commercial panels) or using a field spectrometer (for DIY panels). Save these values in
   a `panel_properties.json` file.
2. **Print an AprilTag for each panel**
3. **Field Setup**
   <br>Position the calibration panels and place the AprilTags according to the placement guide.
4. **Fly your drone mission, capturing images.**
6. **Run reflectdetect**:
   <br>Convert the images to reflectance data.

### Key Concepts and Vocabulary

- **Panel**: Calibration sheets placed in the field, used to compare captured intensity values in images with known
  reflectance values in the `panel_properties.json` file.

- **AprilTag / Tag**: A visual marker used for accurate detection in images.

- **Field**: The area captured by drone imagery.

- **Image**: Individual bands of captured data. reflectdetect assumes all images were taken at equal time intervals and
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

### Installing ExifTool

ExifTool is an essential tool for working with image metadata in reflectdetect. Follow the instructions below to install
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
    - Open the Start Menu, search and select `Edit environment variables for your account`.
    - Under the User variables section, find the variable named `PATH` and select it. Then click Edit.
    - In the new window, click New and enter the path to the folder where you moved exiftool.exe (e.g., `C:\ExifTool`).
    - Click OK to close all the windows.

5. **Verify Installation**:
    - Open Command Prompt (`cmd`) and type `exiftool`. You should see ExifTool's help information, confirming it is
      installed and recognized by your system.

#### macOS Installation

1. **Download ExifTool**:
    - Visit the [ExifTool Home Page](https://exiftool.org) and download the latest MacOS Package (`ExifTool-xx.xx.pkg`).

2. **Install ExifTool**:
    - Double-click the downloaded `.pkg` file and follow the installation instructions.

3. **Verify Installation**:
    - Open Terminal and type `exiftool`. You should see ExifTool's help information, confirming it is installed and
      recognized by your system.

#### Linux Installation

1. **Install via Package Manager**:
    - On most Linux distributions, you can install ExifTool using the package manager. For example, on Ubuntu, run:
      ```sh
      sudo apt-get install libimage-exiftool-perl
      ```
2. **Manual Install**:
    - For manual installation instruction visit the [ExifTool Install Page](https://exiftool.org/install.html#Unix)

3. **Verify Installation**:
    - Open your terminal and type `exiftool`. You should see ExifTool's help information, confirming it is installed and
      recognized by your system.

### Installing reflectdetect

To install the reflectdetect CLI tools to your system, open a command line or terminal and run

```bash
pip install reflectdetect
```

Now the reflectdetect CLI Tools `reflectdetect-converter`, `reflectdetect-apriltag` and `reflectdetect-geolocation`
should be available.

# Usage

Find the information about running each workflow in their respective files:

- [Workflow 1: Geolocation-Based Calibration](workflow1.md)
- [Workflow 2: Apriltag-Based Calibration](workflow2.md)

# Planned Features

- [ ] Support for unequal time intervals between images
- [x] Customize parameters on a per panel basis
- [ ] Add dataset verification script
- [x] Remove 5% of outliers in panel intensities value
- [ ] Provide pdfs for apriltag printouts in `apriltag_printouts`

# Contributing

We welcome and encourage contributions! If you'd like to get involved, please feel free to:

- Fork the repository.
- Make your changes.
- Submit a pull request (PR).

We'll review your PR as soon as possible. Thank you for contributing!

<!--# Gallery -->

  <!--TODO: Add a good image of the panels]-->
  <!--TODO: Add a good image of the apriltags]-->
  <!--TODO: Add a good image of the captured images]-->

# AI Usage Card

# References

# License
