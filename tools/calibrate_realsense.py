"""Fixed-intrinsic multical calibration of prepared RealSense ArUco captures."""
import argparse
import json
from pathlib import Path

import numpy as np

from multical.config.runtime import find_board_config, find_camera_images
from multical.io.import_calib import load_calibration
from multical.io.logging import logger, setup_logging
from multical.workspace import Workspace

from prepare_realsense import read_json, write_json


def detection_report(ws, master):
  visible = ws.point_table.valid.all(axis=-1).any(axis=-1)
  return visibility_report(visible, ws.names.camera, master)


def visibility_report(visible, names, master):
  counts = visible.sum(axis=1)
  overlap = visible.astype(int) @ visible.astype(int).T
  visited = {names.index(master)}
  while True:
    reached = visited | {j for i in visited for j in range(len(names)) if overlap[i, j] > 0}
    if reached == visited:
      break
    visited = reached
  return dict(detected_groups=dict(zip(names, map(int, counts))),
              shared_groups={a: {b: int(overlap[i, j]) for j, b in enumerate(names)}
                             for i, a in enumerate(names)},
              disconnected=[name for i, name in enumerate(names) if i not in visited])


def export_extrinsics(calib, names, master):
  calib = calib.with_master(master)
  return {name: np.linalg.inv(pose).tolist()
          for name, pose, valid in zip(names, calib.camera_poses.poses, calib.camera_poses.valid)
          if valid}


def calibrate(image_path, output, master=None, num_threads=4, detect_only=False):
  output = Path(output).resolve()
  if output.exists():
    raise FileExistsError(f'Refusing to overwrite {output}')
  if num_threads < 1:
    raise ValueError('num_threads must be positive')
  previous_handlers = set(logger.handlers)
  previous_level, previous_propagate = logger.level, logger.propagate
  try:
    return _calibrate(image_path, output, master, num_threads, detect_only)
  except Exception as error:
    if output.exists():
      report_file = output / 'report.json'
      report = read_json(report_file) if report_file.exists() else {}
      report.update(status='failed', error=str(error))
      write_json(report_file, report)
    raise
  finally:
    for handler in list(logger.handlers):
      if handler not in previous_handlers:
        logger.removeHandler(handler)
        handler.close()
    logger.setLevel(previous_level)
    logger.propagate = previous_propagate


def _calibrate(image_path, output, master, num_threads, detect_only):
  root, output = Path(image_path).resolve(), Path(output).resolve()
  conversion = read_json(root / 'conversion.json')
  if conversion['status'] != 'complete':
    raise ValueError('Conversion is incomplete')
  master = master or conversion['master_serial']
  prior = load_calibration(str(root / 'intrinsics.json'))
  if master not in prior.cameras:
    raise ValueError(f'Master serial not present: {master}')
  if output.exists():
    raise FileExistsError(f'Refusing to overwrite {output}')
  output.mkdir(parents=True)
  write_json(output / 'report.json', dict(status='running', master_serial=master))
  ws = Workspace(str(output), 'calibration')
  setup_logging('INFO', [ws.log_handler], log_file=str(output / 'calibration.log'))
  images = find_camera_images(str(root), cameras=sorted(prior.cameras))
  if not images.image_names:
    raise ValueError('No shared images in prepared directory')
  ws.add_camera_images(images, j=num_threads)
  ws.detect_boards(find_board_config(str(root)), load_cache=False, j=num_threads)
  ws.set_calibration(prior.cameras)
  report = detection_report(ws, master)
  report.update(status='detection_only', master_serial=master,
                warning='Single-marker planar poses can be ambiguous. Low reprojection error is not a guarantee of correct extrinsics.')
  write_json(output / 'report.json', report)
  print(json.dumps(report, indent=2))
  if detect_only:
    return report
  if report['disconnected'] or not report['detected_groups'][master]:
    raise ValueError('Detection graph is disconnected; see report.json. Capture more shared views; no extrinsics exported.')
  ws.initialise_poses(exclude_bad_poses=True, pose_error_limit=2.0)
  # Recheck after pose-error rejection: a detected marker can still have no valid pose.
  valid = ws.pose_table.valid.any(axis=-1)
  pose_report = visibility_report(valid, ws.names.camera, master)
  if pose_report['disconnected'] or not pose_report['detected_groups'][master]:
    raise ValueError('Valid-pose graph is disconnected after pose rejection; no extrinsics exported')
  calib = ws.calibrate(cameras=False, boards=False, board_poses=False,
                       camera_poses=True, motion=True, loss='soft_l1')
  if not np.all(calib.camera_poses.valid):
    raise ValueError('Some cameras have no solution; no extrinsics exported')
  # Corner outlier rejection must not remove the links constraining a camera.
  inlier_visibility = (calib.inliers.sum(axis=-1) >= 4).any(axis=-1)
  inlier_report = visibility_report(inlier_visibility, ws.names.camera, master)
  if inlier_report['disconnected'] or not inlier_report['detected_groups'][master]:
    raise ValueError('Inlier observation graph is disconnected; no extrinsics exported')
  # Do not change multical's historical pose format; provide an explicit inverse export.
  transforms = export_extrinsics(calib, ws.names.camera, master)
  if not np.all(np.isfinite(list(transforms.values()))):
    raise ValueError('Non-finite camera transforms; no extrinsics exported')
  errors = np.asarray(calib.reprojection_inliers)
  if not errors.size or not np.all(np.isfinite(errors)):
    raise ValueError('Optimization produced no finite inlier residuals')
  report.update(status='complete', reprojection_inlier_rms_px=float(np.sqrt(np.mean(errors**2))),
                reprojection_inlier_max_px=float(errors.max()), inlier_observations=inlier_report)
  ws.export(master=master)
  ws.dump()
  write_json(output / 'extrinsics.json', dict(
      master_serial=master, translation_unit='meter',
      convention='p_master = T_master_from_camera @ p_camera (homogeneous column vectors)',
      T_master_from_camera=transforms))
  write_json(output / 'report.json', report)
  return report


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('image_path')
  parser.add_argument('--output', required=True)
  parser.add_argument('--master', help='Reference device SERIAL; default from conversion.json')
  parser.add_argument('--num-threads', type=int, default=4)
  parser.add_argument('--detect-only', action='store_true')
  args = parser.parse_args()
  if args.num_threads < 1:
    parser.error('--num-threads must be positive')
  calibrate(args.image_path, args.output, args.master, args.num_threads, args.detect_only)


if __name__ == '__main__':
  main()
