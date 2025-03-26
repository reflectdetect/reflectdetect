<br />
<p align="center">
<a><img src="https://raw.githubusercontent.com/reflectdetect/reflectdetect/main/images/reflectdetect_logo.png" alt="reflectdetect" width="250" height="250" title="reflectdetect" style="border-radius: 50%;"></a>
</p>
<h3 align="center">ReflectDetect</h3>
<p align="center">    An innovative, fully automated command line software for in-flight radiometric calibration of UAV-mounted 2D snapshot multi-camera imagery<br /></p>
<p align="center">
    <img alt="Coverage" src="https://img.shields.io/badge/Coverage-31%25-brightgreen"/>
  <a href="https://github.com/reflectdetect/reflectdetect/blob/main/LICENSE"><img src="https://img.shields.io/github/license/reflectdetect/reflectdetect" alt="License"></a>
  <a href="https://github.com/reflectdetect/reflectdetect/network/members"><img src="https://img.shields.io/github/forks/reflectdetect/reflectdetect?style=social" alt="GitHub forks"></a>
  <a href="https://github.com/reflectdetect/reflectdetect/stargazers"><img src="https://img.shields.io/github/stars/reflectdetect/reflectdetect?style=social" alt="GitHub stars"></a>
<a href="https://doi.org/10.5281/zenodo.14184975"><img src="https://zenodo.org/badge/792282783.svg" alt="DOI"></a>
</p>
<p>
<a href="https://github.com/reflectdetect/reflectdetect/issues">Report Bug</a>
·
<a href="https://github.com/reflectdetect/reflectdetect/issues">Request Feature</a>
</p>
<p>
<a href="https://reflectdetect.readthedocs.io">Documentation</a>
</p>
<a href="https://ifzgoettingen-my.sharepoint.com/:f:/g/personal/heim_ifz-goettingen_de/EmSDi-poSitJpDbd4Xrrhj8BF-9x5LMSKSMOvJbj5OqmXg?e=g0oss6">Download Example Datasets</a>
<a href="https://github.com/reflectdetect/reflectdetect/blob/main/README.md#installation">How to install?</a>



## Overview

Welcome to the official repository for the paper, "ReflectDetect: A software tool for AprilTag-guided in-flight radiometric calibration for UAV snapshot multi-cameras".

<!-- TODO: Properly cite paper -->

### Abstract
> Unmanned Aerial Vehicles (UAVs) equipped with optical sensors have transformed remote sensing in vegetation science by providing high-resolution,
> on-demand data, enhancing studies in forestry, agriculture, and environmental monitoring. However, accurate radiometric calibration of UAV imagery
> remains challenging. A common practice, using a single calibration target while holding the UAV-mounted camera close above it, is being criticized
> as the hemisphere is invisibly shaded and the reference images are not collected under flight conditions. ReflectDetect addresses these challenges
> by allowing in-flight radiometric calibration through automated detection via two different approaches: 1) a geotagging approach leveraging
> high-precision coordinates of the reflectance targets and 2) AprilTag-based detection, a visual fiducial system frequently used in robotics.
> ReflectDetect is available through a user-friendly command-line interface and open-source. It now enables users to design new in-flight calibration
> studies to eventually improve radiometric calibration in applied UAV remote sensing.

### Workflow 1: Geolocation-Based Calibration

1. **Know your calibration panel reflectance factors**  
For each of your calibration panel, obtain the reflectance factor and ensure that it has a flat response according to the bands of your camera system. This information can be obtain from the manufacturer (for commercial panels) or by using a field spectrometer (for DIY panels). Save these values in a `panel_properties.json` file (see [example data](https://ifzgoettingen-my.sharepoint.com/:f:/g/personal/heim_ifz-goettingen_de/EmSDi-poSitJpDbd4Xrrhj8BF-9x5LMSKSMOvJbj5OqmXg?e=g0oss6)).
In case you would like to build your own calibration panels, the following publication provides basic information on that topic:
    > C. Wang and S. W. Myint, "A Simplified Empirical Line Method of Radiometric Calibration for Small Unmanned Aircraft Systems-Based Remote Sensing," in IEEE Journal of Selected Topics in Applied Earth >  Observations and Remote Sensing, vol. 8, no. 5, pp. 1876-1885, May 2015, doi: [10.1109/JSTARS.2015.2422716](https://doi.org/10.1109/JSTARS.2015.2422716).
2. **Position the calibration panels in the field**
During placement, avoid that the panel(s) is/are shaded by adjacent plant or objects. Placing the panels in the center of your mapping area will increase the number of images that contain the panels.
3. **Capture the locations of the panel corners**  
We used an [Ardusimple RTK Calibrated Surveyor Kit](https://www.ardusimple.com/product/rtk-calibrated-surveyor-kit/) in our testing, save the data in a `panel_locations.gpkg` file (see [example data](https://ifzgoettingen-my.sharepoint.com/:f:/g/personal/heim_ifz-goettingen_de/EmSDi-poSitJpDbd4Xrrhj8BF-9x5LMSKSMOvJbj5OqmXg?e=g0oss6)).
4. **Fly your drone mission, capturing images.**
5. **Rectifying and geo-reference the images**
   <br>We used [Metashape](https://www.agisoft.com/) in our testing
6. **Run reflectdetect**:
   <br>Convert the orthophotos to reflectance data.

### Workflow 2: AprilTag-Based Calibration

1. **Gather Panel Reflectance Data**  
For each of your calibration panel, obtain the reflectance factor and ensure that it has a flat response according to the bands of your camera system. This information can be obtain from the manufacturer (for commercial panels) or by using a field spectrometer (for DIY panels). Save these values in a `panel_properties.json` file (see [example data](https://ifzgoettingen-my.sharepoint.com/:f:/g/personal/heim_ifz-goettingen_de/EmSDi-poSitJpDbd4Xrrhj8BF-9x5LMSKSMOvJbj5OqmXg?e=g0oss6)).
2. **Print an AprilTag for each panel, see [Official AprilTag Repo](https://github.com/AprilRobotics/apriltag-imgs/tree/master/tag25h9)**
3. **Field Setup**
   <br>Position the calibration panels and place the AprilTags according to the [placement guide](#gallery).
4. **Fly your drone mission, capturing images.**
5. **Run reflectdetect**:
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
## Docker Installation
[Install](https://docs.docker.com/engine/install/) Docker (for example using [Docker Desktop](https://docs.docker.com/desktop/))

Build the 3 different CLI tools using docker:
- `docker build -t reflectdetect-apriltag-image -f Dockerfile_apriltag .`
- `docker build -t reflectdetect-converter-image -f Dockerfile_converter .`
- `docker build -t reflectdetect-geolocation-image -f Dockerfile_geolocation .`

Run a CLI tool using docker after building:
- `docker run -v /path/to/your/dataset:/data reflectdetect-apriltag-image`
- `docker run -v /path/to/your/dataset:/data reflectdetect-apriltag-image`
- `docker run -v /path/to/your/dataset:/data reflectdetect-apriltag-image`

## Manual Installation

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
- [ ] Allow running the calibration across multiple flight missions.
- [ ] Add absolute calibration functions and adjustments for more camera manufacturers. 

# Contributing

We welcome and encourage contributions! If you'd like to get involved, please feel free to:

- Fork the repository.
- Make your changes.
- Submit a pull request (PR).

We'll review your PR as soon as possible. Thank you for contributing!

# Gallery
### Apriltag Placement Guide

<img src="./images/placement_small.png" alt="AprilTag Placement" title="Align mid-sections of each calibration panel and AprilTag. We kept the tag family label at the opposite side of the calibration panel." style="width:50%; height:auto;">

  <!--TODO: Add a good image of the panels]-->
  <!--TODO: Add a good image of the apriltags]-->
  <!--TODO: Add a good image of the captured images]-->

# AI Usage Card

We used AI tools as follows: [AI Usage Card](https://ifzgoettingen-my.sharepoint.com/:b:/g/personal/heim_ifz-goettingen_de/Ed4LdB0RuX9OmMEGOajw_TsBSw8e9H565nfjgPqyUYP6Fw?e=XL8usx) 

# References

1. Aasen H, Honkavaara E, Lucieer A, Zarco-Tejada P. Quantitative Remote Sensing at Ultra-High Resolution with UAV Spectroscopy: A Review of Sensor Technology, Measurement Procedures, and Data Correction Workflows. Remote Sens 2018;10:1091. https://doi.org/10.3390/rs10071091.
2.	Maes WH, Steppe K. Perspectives for Remote Sensing with Unmanned Aerial Vehicles in Precision Agriculture. Trends Plant Sci 2019;24:152–64. https://doi.org/10.1016/j.tplants.2018.11.007.
3.	Daniels L, Eeckhout E, Wieme J, Dejaegher Y, Audenaert K, Maes WH. Identifying the Optimal Radiometric Calibration Method for UAV-Based Multispectral Imaging. Remote Sens 2023;15:2909. https://doi.org/10.3390/rs15112909.
4.	Wang C, Myint SW. A Simplified Empirical Line Method of Radiometric Calibration for Small Unmanned Aircraft Systems-Based Remote Sensing. IEEE J Sel Top Appl Earth Obs Remote Sens 2015;8:1876–85. https://doi.org/10.1109/JSTARS.2015.2422716.
5.	Cao S, Danielson B, Clare S, Koenig S, Campos-Vargas C, Sanchez-Azofeifa A. Radiometric calibration assessments for UAS-borne multispectral cameras: Laboratory and field protocols. ISPRS J Photogramm Remote Sens 2019;149:132–45. https://doi.org/10.1016/j.isprsjprs.2019.01.016.
6.	Fawcett D, Anderson K. Investigating impacts of calibration methodology and irradiance variations on lightweight drone-based sensor derived surface reflectance products. In: Neale CM, Maltese A, editors. Remote Sens. Agric. Ecosyst. Hydrol. XXI, Strasbourg, France: SPIE; 2019, p. 13. https://doi.org/10.1117/12.2533106.
7.	Chakhvashvili E, Siegmann B, Bendig J, Rascher U. Comparison of Reflectance Calibration Workflows for a UAV-Mounted Multi-Camera Array System. 2021 IEEE Int. Geosci. Remote Sens. Symp. IGARSS, Brussels, Belgium: IEEE; 2021, p. 8225–8. https://doi.org/10.1109/IGARSS47720.2021.9555143.
8.	Eltner A, Hoffmeister D, Kaiser A, Karrasch P, Klingbeil L, Stöcker C, et al., editors. UAVs for the environmental sciences: methods and applications. Darmstadt: wbg Academic; 2022.
9.	Ban S, Kim T. AUTOMATED REFLECTANCE TARGET DETECTION FOR AUTOMATED VICARIOUS RADIOMETRIC CORRECTION OF UAV IMAGES. Int Arch Photogramm Remote Sens Spat Inf Sci 2021;XLIII-B1-2021:133–7. https://doi.org/10.5194/isprs-archives-XLIII-B1-2021-133-2021.
10.	Wang J, Olson E. AprilTag 2: Efficient and robust fiducial detection. 2016 IEEERSJ Int. Conf. Intell. Robots Syst. IROS, 2016, p. 4193–8. https://doi.org/10.1109/IROS.2016.7759617.
11.	Kalaitzakis M, Cain B, Carroll S, Ambrosi A, Whitehead C, Vitzilaios N. Fiducial Markers for Pose Estimation. J Intell Robot Syst 2021;101:71. https://doi.org/10.1007/s10846-020-01307-9.
12.	Papoutsoglou EA, Athanasiadis IN, Visser RGF, Finkers R. The benefits and struggles of FAIR data: the case of reusing plant phenotyping data. Sci Data 2023;10:457. https://doi.org/10.1038/s41597-023-02364-z.
13.	Manzano S, Julier ACM. How FAIR are plant sciences in the twenty-first century? The pressing need for reproducibility in plant ecology and evolution. Proc R Soc B Biol Sci 2021;288:20202597. https://doi.org/10.1098/rspb.2020.2597.
14.	Barker M, Chue Hong NP, Katz DS, Lamprecht A-L, Martinez-Ortiz C, Psomopoulos F, et al. Introducing the FAIR Principles for research software. Sci Data 2022;9:622. https://doi.org/10.1038/s41597-022-01710-x.
15.	Grünwald NJ, Bock CH, Chang JH, De Souza AA, Ponte EMD, du Toit LJ, et al. Open Access and Reproducibility in Plant Pathology Research: Guidelines and Best Practices. Phytopathology® 2024:PHYTO-12-23-0483-IA. https://doi.org/10.1094/PHYTO-12-23-0483-IA.
16.	Reichman OJ, Jones MB, Schildhauer MP. Challenges and Opportunities of Open Data in Ecology. Science 2011;331:703–5. https://doi.org/10.1126/science.1197962.
17.	Serwadda D, Ndebele P, Grabowski MK, Bajunirwe F, Wanyenze RK. Open data sharing and the Global South—Who benefits? Science 2018;359:642–3. https://doi.org/10.1126/science.aap8395.
18.	Wilkinson MD, Dumontier M, Aalbersberg IjJ, Appleton G, Axton M, Baak A, et al. The FAIR Guiding Principles for scientific data management and stewardship. Sci Data 2016;3:160018. https://doi.org/10.1038/sdata.2016.18.


# License
