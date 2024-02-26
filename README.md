# detect_markers

## Usage

### Calibration

```shell
python3 -m detect_markers calibrate [src] [dst]
```

`src`: Input movie file, e.g. `00001.MTS`.

`dst`: Output calibration parameters file, e.g. `camera_params.yaml`.

### Detection

```shell
python3 -m detect_markers detect [src] [dst] [camera_params] --marker-ids [marker_ids]
```

`src`: Input movie file, e.g. `00001.MTS`.

`dst`: Output marker detection result file, e.g. `out.csv`.

`camera_params`: Camera parameters file generated by calibration, e.g. `camera_params.yaml`

`marker_ids`: Marker IDs to detect, e.g. `0 1 2 3`
