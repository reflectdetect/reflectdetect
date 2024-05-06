# reflectdetect
Automatic detection of reflectance calibration panels in multiband drone imagery

### Planning Pad
https://pad.gwdg.de/_OcW4npSSvW9wHiAjN016Q#

### Parameters supplied by the user
```
panel_amount: int
flight_height: float
panel:
  reflectance: [{from, to, reflectance}]
  panel_size : float
  panel_shape: RECT | CIRCLE
image:
  lens_characteristics
  band_number: int
  id: int
```
### Files created by the script
Given `data/image_0001_1.tif` the script created multiple files 
- `data/output/metadata/image_0001_1_detections.json`
- `data/output/metadata/image_0001_1_panels.json`
- `data/output/image_0001_1_transformed.tif`

#### `*_detections.json`:

```
[
{x, y}
]
```