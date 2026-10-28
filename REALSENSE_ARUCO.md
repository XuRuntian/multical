# RealSense: Fixed-Intrinsic ArUco Extrinsics

This workflow uses the existing sync-lab export and multical bundle adjustment.
It keeps the RGB intrinsics and distortion coefficients read from each device
fixed. It does not recalibrate depth, RGB intrinsics, or device-internal poses.

Defaults:

- Reference camera serial: `338522300613` (currently `cam03`).
- Marker: ArUco `6X6_50`, ID `0`, black outer square side `0.1` meters.
- Output camera names are **serial numbers**, not session-local `camXX` names.
- Images remain in their original distorted pixel coordinates. Inverse
  Brown-Conrady uses a separate camera model, matched against librealsense's
  `rs2_deproject_pixel_to_point`; coefficients are not treated as OpenCV Brown
  coefficients. None and ordinary Brown-Conrady are also supported. Other
  distortion models are rejected rather than guessed.

## Capture

Keep all six cameras fixed throughout all sessions. For each marker pose:

1. Place the marker flat and rigid, at a new position/orientation.
2. Let it stop moving before recording a short session (e.g. 1-2 seconds).
3. Use a new output directory each time. Keep the marker still until capture ends.

Example in the capture repository:

```bash
cd /home/user/realsense-sync-lab
.venv/bin/python -m sync_lab record \
  --config configs/hardware_6cam.json \
  --output data/aruco_pose_001 --session-id aruco_pose_001 --duration 2
```

Repeat with `aruco_pose_002`, etc. Collect dozens of diverse poses rather than
many duplicate frames of one pose. Include different tilts, distances, and
locations. Ensure the tag is large/sharp enough in the image. A single planar
marker has pose ambiguity, especially when small or nearly frontal; inspect the
result against the actual rig, not only its reprojection error.

Alternatively, newer sync-lab checkouts can keep all cameras streaming and
capture many numbered stationary poses in one interactive session:

```bash
cd /home/user/realsense-sync-lab
.venv/bin/python -m sync_lab capture-calibration \
  --config configs/hardware_6cam.json \
  --output data/rig_calibration_001 --settle 0.5 --burst 1.0
```

Press Enter once per stationary pose and type `q` after the last pose. This
format is handled by the separate `prepare_segmented_realsense.py` script below;
it is deliberately not mixed into the legacy HO-Cap preparation script.

Each camera must share tag observations with others in some synchronized groups;
the observation graph must connect all cameras to the reference. All six need
not see the marker simultaneously. Only one physical instance of ID 0 may be
visible at a time. Do not merge recordings after moving any camera. Do not change
resolution or stream profile between captures; inconsistent intrinsics are rejected.

`record` creates `hocap/export_manifest.json` and `hocap/frames.jsonl`. If only
raw data exists, first use sync-lab's `export-hocap`. The converter does not
pair raw files by their names or invent synchronization.

## Prepare And Merge

```bash
cd /home/user/1/multical
conda activate multical-py
taskset -c 0-7 python tools/prepare_realsense.py \
  /home/user/realsense-sync-lab/data \
  --output ./aruco_inputs
```

Pass a parent directory to discover exports in its immediate child directories
(e.g. `session_001/hocap`, `session_02/hocap`), in natural name order. Unrelated
folders are ignored; failed/incomplete exports and raw-only sessions cause an
error rather than being silently skipped. The command prints included exports.
Use a dedicated calibration data directory: do not mix captures from different
camera arrangements or unrelated recordings. You can still pass individual
session/export paths, or multiple input paths. Duplicate exports are rejected.

Default: select the middle exported synchronized group from **each** session.
Use `--frames-per-session N` for N evenly spaced groups, or `0` for all groups
(e.g. when a session contains multiple stationary poses). The middle frame is
not guaranteed to have the best visibility; run detection and select more groups
if necessary. Inputs can also be `hocap` directories directly.

The converter validates complete exports and camera identity/profile, excludes
depth images, and renames each selected group to a unique `pose_NNNNNN.jpg`
shared across serial-number directories. It uses absolute symlinks, so keep the
source captures in place. It never overwrites an existing output directory.

By default (`--intrinsics-policy strict`), numerical intrinsics must match.
If device reads vary slightly between sessions and you explicitly accept the
approximation, add `--intrinsics-policy first`. Each camera then uses the first
session's K and distortion coefficients for all images (no compensating image
rectification). Parent-directory scans use natural session-name order. Size and
distortion model must still match. The command warns about differences and
`conversion.json` preserves reference exports, source parameters, and deltas.
Source captures are never modified. Validate resulting geometry on held-out views;
small focal-length differences do not guarantee small extrinsic errors.

### Prepare an interactive segmented capture

Use this command only for `sync_lab capture-calibration` output. Every pose in
`calibration_images/frames.jsonl` is included; there is no middle-frame or
`--frames-per-session` selection because sync-lab already selected one valid
synchronized color group per pose.

```bash
cd /home/user/multical
taskset -c 0-7 python tools/prepare_segmented_realsense.py \
  /home/user/realsense-sync-lab/data/rig_calibration_001 \
  --output ./aruco_inputs
```

The input may also be the `calibration_images` directory itself or a parent
whose immediate children contain segmented captures. Old HO-Cap sessions are
ignored by this new script; `prepare_realsense.py` remains the independent
legacy converter. PNG images remain lossless PNG symlinks in the prepared data.

A single interactive capture contains one device-intrinsics snapshot, so use
the default `--intrinsics-policy strict`. Only add `--intrinsics-policy first`
when combining multiple independent segmented captures whose numerical device
intrinsics differ slightly and that approximation is explicitly acceptable.

Output:

- `<serial>/pose_NNNNNN.jpg`: links to RGB images only.
- `intrinsics.json`: unchanged device RGB calibration, in multical format.
- `boards.yaml`: single-marker configuration.
- `conversion.json`: reference serial, source calibrations and frame provenance.

## Detect And Calibrate

```bash
taskset -c 0-7 python tools/calibrate_realsense.py ./aruco_inputs \
  --output ./aruco_detection --detect-only

taskset -c 0-7 python tools/calibrate_realsense.py ./aruco_inputs \
  --output ./aruco_results
```

Optional `--master SERIAL` overrides the prepared reference; it does not change
hardware synchronization. Detection reports per-camera counts, shared-group
counts, and cameras disconnected from the master. Full calibration refuses to
export an incomplete rig and rechecks connectivity after pose and outlier rejection.

Successful output includes:

- `extrinsics.json`: use **`T_master_from_camera`**. For homogeneous column
  vectors: `p_master = T_master_from_camera @ p_camera`. Translation is in meters.
  The master's transform is identity. Axes are the RGB camera axes, not a
  table-aligned world frame.
- `calibration.json`: native multical format. Its historical pose names are
  misleading; do not confuse these with the explicitly directed transforms above.
- `calibration.pkl`: workspace for visualization.
- `report.json`: completion status, connectivity and inlier reprojection errors.
- `calibration.log` and detection cache.

For depth points, first use each device's depth-to-RGB calibration before applying
the RGB-camera-to-master transform. Depth images aligned to RGB use the exported
aligned-depth intrinsics; do not use the original depth-camera intrinsics on them.

## Verification

```bash
taskset -c 0-7 python -m unittest discover -s tests -v
```

Tests cover marker detection/pose, SDK deprojection reference values, camera
serialization, multi-session identity matching, frame selection, validation, and
synthetic fixed-intrinsic bundle adjustment with explicit transform direction.
Real captures still require physical validation of geometry and held-out poses.
