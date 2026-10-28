"""Prepare multical inputs from sync-lab interactive segmented captures."""

import argparse
import json
from pathlib import Path
import re
import shutil

import cv2
import numpy as np


DEFAULT_MASTER = '338522300613'


def natural_key(path):
  return [int(part) if part.isdigit() else part.lower()
          for part in re.split(r'(\d+)', Path(path).name)]


def read_json(path):
  return json.loads(Path(path).read_text())


def write_json(path, data):
  Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def import_intrinsics(intrinsics):
  model = intrinsics['model'].removeprefix('distortion.')
  models = {'none': 'standard', 'brown_conrady': 'standard',
            'inverse_brown_conrady': 'realsense_inverse_brown_conrady'}
  if model not in models:
    raise ValueError(f'Unsupported RealSense distortion model: {model}')
  fx, fy, cx, cy = [intrinsics[k] for k in ('fx', 'fy', 'ppx', 'ppy')]
  coeffs = intrinsics['coeffs']
  if (fx <= 0 or fy <= 0 or len(coeffs) != 5 or
      not np.all(np.isfinite([fx, fy, cx, cy, *coeffs]))):
    raise ValueError('Invalid camera intrinsics')
  size = [intrinsics['width'], intrinsics['height']]
  if any(type(n) is not int or n <= 0 for n in size):
    raise ValueError('Invalid image dimensions')
  return dict(model=models[model], image_size=size,
              K=[[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
              dist=[0.0] * 5 if model == 'none' else coeffs)


def same_camera(a, b):
  return (a['model'] == b['model'] and a['image_size'] == b['image_size']
          and np.allclose(a['K'], b['K'], atol=1e-7, rtol=0)
          and np.allclose(a['dist'], b['dist'], atol=1e-9, rtol=0))


def segmented_export_for(directory):
  if (directory / 'manifest.json').is_file():
    manifest = read_json(directory / 'manifest.json')
    if manifest.get('schema') == 'sync_lab.calibration_images.v1':
      return directory
  candidate = directory / 'calibration_images'
  if (candidate / 'manifest.json').is_file():
    return candidate
  if (directory / 'manifest.json').is_file():
    manifest = read_json(directory / 'manifest.json')
    if manifest.get('capture_mode') == 'segmented':
      raise ValueError(f'Segmented capture has no complete calibration_images export: {directory}')
  return None


def discover_segmented(inputs):
  """Accept calibration_images, capture sessions, or one-level parent directories."""
  exports = []
  for value in inputs:
    directory = Path(value).resolve()
    if not directory.is_dir():
      raise ValueError(f'Input directory does not exist: {directory}')
    direct = segmented_export_for(directory)
    if direct is not None:
      exports.append(direct)
      continue
    children = [segmented_export_for(child) for child in
                sorted(directory.iterdir(), key=natural_key) if child.is_dir()]
    children = [child for child in children if child is not None]
    if not children:
      raise ValueError(f'No segmented calibration capture found in {directory} or its immediate children')
    exports.extend(children)
  if len(set(exports)) != len(exports):
    raise ValueError('Duplicate segmented calibration capture')
  return exports


def load_segmented(export):
  manifest = read_json(export / 'manifest.json')
  if manifest.get('schema') != 'sync_lab.calibration_images.v1' or manifest.get('status') != 'complete':
    raise ValueError(f'Calibration image export is not complete: {export}')
  capture = Path(manifest['source_capture']).resolve()
  capture_manifest = read_json(capture / 'manifest.json')
  if (capture_manifest.get('capture_mode') != 'segmented'
      or capture_manifest.get('status') not in ('complete', 'interrupted')
      or capture_manifest.get('errors')):
    raise ValueError(f'Invalid source segmented capture: {capture}')
  ids = manifest['camera_ids']
  if not ids or len(set(ids.values())) != len(ids) or any(not serial.isdigit() for serial in ids):
    raise ValueError(f'Invalid camera identities: {export}')
  configured = {device['serial']: device.get('id', device['serial'])
                for device in capture_manifest['config']['devices']}
  if configured != ids or set(ids) != set(capture_manifest['devices']):
    raise ValueError(f'Calibration export and source capture identities differ: {export}')

  cameras, calibration = {}, {}
  for serial, camera_id in ids.items():
    if Path(camera_id).name != camera_id or '\\' in camera_id:
      raise ValueError(f'Invalid camera ID: {camera_id}')
    original = capture_manifest['devices'][serial]['calibration']
    cameras[serial] = import_intrinsics(original['intrinsics']['color'])
    calibration[serial] = original

  rows = [json.loads(line) for line in (export / 'frames.jsonl').read_text().splitlines() if line.strip()]
  if not rows or len(rows) != manifest['exported_poses']:
    raise ValueError(f'Empty/incomplete calibration pose index: {export}')
  pose_ids = [row['pose_index'] for row in rows]
  if (any(type(pose) is not int or pose < 0 for pose in pose_ids)
      or len(set(pose_ids)) != len(pose_ids)):
    raise ValueError(f'Pose indices must be unique nonnegative integers: {export}')
  frames = []
  for row in rows:
    if set(row['cameras']) != set(ids.values()):
      raise ValueError(f'Pose has missing/extra cameras: {export}')
    paths = {}
    for serial, camera_id in ids.items():
      frame = row['cameras'][camera_id]
      if frame['serial'] != serial:
        raise ValueError('Pose camera serial mismatch')
      source = (export / frame['path']).resolve()
      if export not in source.parents or source.suffix.lower() != '.png' or not source.is_file():
        raise ValueError(f'Invalid calibration image path: {source}')
      image = cv2.imread(str(source), cv2.IMREAD_COLOR)
      if image is None or list(image.shape[1::-1]) != cameras[serial]['image_size']:
        raise ValueError(f'Unreadable image or profile mismatch: {source}')
      paths[serial] = str(source)
    frames.append(dict(export=str(export), source_capture=str(capture),
                       source_pose_index=row['pose_index'], source_segment_id=row['segment_id'],
                       source_group=row, paths=paths))
  return dict(export=export, capture=capture, ids=ids, cameras=cameras,
              calibration=calibration, frames=frames)


def prepare(sessions, output, master=DEFAULT_MASTER, intrinsics_policy='strict'):
  output = Path(output).resolve()
  if output.exists():
    raise FileExistsError(f'Refusing to overwrite {output}')
  if intrinsics_policy not in ('strict', 'first'):
    raise ValueError('intrinsics_policy must be strict or first')

  cameras, references, source_calibration, overrides, selected = {}, {}, {}, [], []
  expected_serials = None
  for export in discover_segmented(sessions):
    session = load_segmented(export)
    current = session['cameras']
    if expected_serials is None:
      expected_serials = set(current)
    elif set(current) != expected_serials:
      raise ValueError('All segmented captures must contain the same camera serials')
    for serial, camera in current.items():
      if serial in cameras and not same_camera(camera, cameras[serial]):
        reference = cameras[serial]
        if camera['model'] != reference['model'] or camera['image_size'] != reference['image_size']:
          raise ValueError(f'Image size/distortion model changed for {serial}; cannot override these profiles')
        if intrinsics_policy == 'strict':
          raise ValueError(f'Intrinsics changed for {serial}; use --intrinsics-policy first only to explicitly reuse the first capture calibration')
        overrides.append(dict(serial=serial, export=str(export), reference_export=references[serial],
                              K_delta=(np.array(camera['K']) - np.array(reference['K'])).tolist(),
                              dist_delta=(np.array(camera['dist']) - np.array(reference['dist'])).tolist()))
      if serial not in cameras:
        cameras[serial], references[serial] = camera, str(export)
      source_calibration.setdefault(serial, []).append(dict(
        export=str(export), capture=str(session['capture']), calibration=session['calibration'][serial]))
    selected.extend(session['frames'])

  if not selected or master not in cameras:
    raise ValueError(f'No poses, or master serial {master} is missing')
  output.mkdir(parents=True)
  conversion = dict(schema='multical.segmented_realsense_input.v1', status='building',
                    master_serial=master, camera_names='device serial numbers',
                    intrinsics_policy=intrinsics_policy, intrinsics_reference_exports=references,
                    intrinsics_overrides=overrides, source_calibration=source_calibration, frames=[])
  write_json(output / 'conversion.json', conversion)
  try:
    for serial in cameras:
      (output / serial).mkdir()
    for index, frame in enumerate(selected):
      name = f'pose_{index:06d}.png'
      for serial, source in frame['paths'].items():
        (output / serial / name).symlink_to(source)
      conversion['frames'].append(dict(filename=name, **frame))
    write_json(output / 'intrinsics.json', dict(cameras=cameras))
    board = Path(__file__).resolve().parents[1] / 'example_boards/aruco_100mm.yaml'
    shutil.copyfile(board, output / 'boards.yaml')
    conversion['status'] = 'complete'
    write_json(output / 'conversion.json', conversion)
  except BaseException:
    conversion['status'] = 'failed'
    write_json(output / 'conversion.json', conversion)
    raise
  return conversion


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('sessions', nargs='+',
                      help='Segmented capture, calibration_images directory, or one-level parent')
  parser.add_argument('--output', required=True)
  parser.add_argument('--master', default=DEFAULT_MASTER, help='Reference device SERIAL, not camXX')
  parser.add_argument('--intrinsics-policy', choices=['strict', 'first'], default='strict',
                      help='strict: require matching captures; first: reuse first capture intrinsics')
  args = parser.parse_args()
  result = prepare(args.sessions, args.output, args.master, args.intrinsics_policy)
  exports = list(dict.fromkeys(frame['export'] for frame in result['frames']))
  for export in exports:
    print(f'Included: {export}')
  if result['intrinsics_overrides']:
    print('WARNING: Reusing first-capture intrinsics; source images are not rectified for differences.')
  print(f"Prepared {len(result['frames'])} poses from {len(exports)} segmented capture(s); "
        f"master={result['master_serial']}; output={args.output}")


if __name__ == '__main__':
  main()
