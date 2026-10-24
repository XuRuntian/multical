"""Merge sync-lab HO-Cap exports into color-only multical inputs."""
import argparse
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
from natsort import natsorted


DEFAULT_MASTER = '338522300613'


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
              dist=[0.0]*5 if model == 'none' else coeffs)


def same_camera(a, b):
  return (a['model'] == b['model'] and a['image_size'] == b['image_size']
          and np.allclose(a['K'], b['K'], atol=1e-7, rtol=0)
          and np.allclose(a['dist'], b['dist'], atol=1e-9, rtol=0))


def discover_exports(inputs):
  """Accept exports, individual sessions, or a parent containing sessions."""
  def export_for(directory):
    if (directory / 'export_manifest.json').is_file():
      return directory
    if (directory / 'hocap').is_dir():
      return directory / 'hocap'
    if (directory / 'raw' / 'manifest.json').is_file():
      raise ValueError(f'Session has only raw data: {directory}; run sync-lab export-hocap first')
    return None

  exports = []
  for value in inputs:
    directory = Path(value).resolve()
    if not directory.is_dir():
      raise ValueError(f'Input directory does not exist: {directory}')
    direct = export_for(directory)
    if direct is not None:
      exports.append(direct)
      continue
    children = []
    for child in natsorted(directory.iterdir(), key=lambda p: p.name):
      if child.is_dir():
        export = export_for(child)
        if export is not None:
          children.append(export)
    if not children:
      raise ValueError(f'No session exports found in {directory} or its immediate subdirectories')
    exports.extend(children)
  return exports


def prepare(sessions, output, master=DEFAULT_MASTER, frames_per_session=1, intrinsics_policy='strict'):
  output = Path(output).resolve()
  if output.exists():
    raise FileExistsError(f'Refusing to overwrite {output}')
  if frames_per_session < 0:
    raise ValueError('frames_per_session must be >= 0 (0 means all groups)')
  if intrinsics_policy not in ('strict', 'first'):
    raise ValueError('intrinsics_policy must be strict or first')
  cameras, source_calibration, selected, roots = {}, {}, [], set()
  reference_exports, overrides = {}, []
  for session in discover_exports(sessions):
    root = session.resolve()
    if root in roots:
      raise ValueError(f'Duplicate session: {root}')
    roots.add(root)
    manifest = read_json(root / 'export_manifest.json')
    if manifest['status'] != 'complete':
      raise ValueError(f'Export is not complete: {root}')
    ids = manifest['camera_ids']  # serial -> local camera ID
    if len(set(ids.values())) != len(ids):
      raise ValueError('Duplicate camera IDs')
    current = {}
    for serial, camera_id in ids.items():
      if Path(camera_id).name != camera_id or not serial.isdigit():
        raise ValueError('Invalid camera name or serial')
      data = read_json(root / camera_id / 'calibration.json')
      if data['serial'] != serial or data['id'] != camera_id:
        raise ValueError(f'Camera identity mismatch: {root / camera_id}')
      current[serial] = import_intrinsics(data['original_calibration']['intrinsics']['color'])
      if serial in cameras and not same_camera(current[serial], cameras[serial]):
        reference = cameras[serial]
        if current[serial]['model'] != reference['model'] or current[serial]['image_size'] != reference['image_size']:
          raise ValueError(f'Image size/distortion model changed for {serial}; cannot override these profiles')
        if intrinsics_policy == 'strict':
          raise ValueError(f'Intrinsics/profile changed for {serial}; use --intrinsics-policy first only to explicitly reuse the first session calibration')
        overrides.append(dict(serial=serial, export=str(root), reference_export=reference_exports[serial],
                              K_delta=(np.array(current[serial]['K'])-np.array(reference['K'])).tolist(),
                              dist_delta=(np.array(current[serial]['dist'])-np.array(reference['dist'])).tolist()))
      source_calibration.setdefault(serial, []).append(dict(export=str(root), calibration=data))
    if cameras and current.keys() != cameras.keys():
      raise ValueError('All sessions must contain the same camera serials')
    for serial, camera in current.items():
      if serial not in cameras:
        cameras[serial] = camera
        reference_exports[serial] = str(root)
    rows = [json.loads(line) for line in (root / 'frames.jsonl').read_text().splitlines() if line.strip()]
    if not rows or len(rows) != manifest['exported_groups']:
      raise ValueError(f'Empty/incomplete export index: {root}')
    if len({r['frame_id'] for r in rows}) != len(rows):
      raise ValueError('Duplicate frame IDs in export index')
    # One stationary pose per session: take the middle complete group by default.
    if frames_per_session == 1:
      rows = [rows[len(rows)//2]]
    elif frames_per_session > 1 and len(rows) > frames_per_session:
      rows = [rows[i] for i in np.linspace(0, len(rows)-1, frames_per_session, dtype=int)]
    for row in rows:
      if set(row['cameras']) != set(ids.values()):
        raise ValueError('Frame group has missing/extra cameras')
      paths = {}
      for serial, camera_id in ids.items():
        frame = row['cameras'][camera_id]
        if frame['serial'] != serial:
          raise ValueError('Frame camera serial mismatch')
        source = (root / frame['color_path']).resolve()
        if root not in source.parents or not source.name.startswith('color_') or source.suffix.lower() != '.jpg':
          raise ValueError(f'Invalid color image path: {source}')
        image = cv2.imread(str(source))
        if image is None or list(image.shape[1::-1]) != cameras[serial]['image_size']:
          raise ValueError(f'Unreadable image or profile mismatch: {source}')
        paths[serial] = str(source)
      selected.append(dict(export=str(root), source_frame_id=row['frame_id'],
                           source_group=row, paths=paths))
  if not selected or master not in cameras:
    raise ValueError(f'No frames, or master serial {master} is missing')
  output.mkdir(parents=True)
  conversion = dict(status='building', master_serial=master,
                    camera_names='device serial numbers', frames_per_session=frames_per_session,
                    intrinsics_policy=intrinsics_policy, intrinsics_reference_exports=reference_exports,
                    intrinsics_overrides=overrides,
                    source_calibration=source_calibration, frames=[])
  write_json(output / 'conversion.json', conversion)
  try:
    for serial in cameras:
      (output / serial).mkdir()
    for i, frame in enumerate(selected):
      name = f'pose_{i:06d}.jpg'
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
  parser.add_argument('sessions', nargs='+', help='Parent data directories, individual sessions, or hocap exports (parent scan is one level)')
  parser.add_argument('--output', required=True)
  parser.add_argument('--master', default=DEFAULT_MASTER, help='Reference device SERIAL, not camXX')
  parser.add_argument('--frames-per-session', type=int, default=1,
                      help='Select N evenly spaced groups; default: middle group only; 0: all')
  parser.add_argument('--intrinsics-policy', choices=['strict', 'first'], default='strict',
                      help='strict: require matching intrinsics; first: explicitly reuse first session intrinsics (size/model must still match)')
  args = parser.parse_args()
  result = prepare(args.sessions, args.output, args.master, args.frames_per_session, args.intrinsics_policy)
  exports = list(dict.fromkeys(frame['export'] for frame in result['frames']))
  for export in exports:
    print(f'Included: {export}')
  if result['intrinsics_overrides']:
    print('WARNING: Reusing first-session intrinsics; source images are NOT rectified to compensate for differences.')
    for serial in sorted({item['serial'] for item in result['intrinsics_overrides']}):
      differences = [item for item in result['intrinsics_overrides'] if item['serial'] == serial]
      max_k = max(float(np.abs(item['K_delta']).max()) for item in differences)
      max_dist = max(float(np.abs(item['dist_delta']).max()) for item in differences)
      print(f'  {serial}: {len(differences)} differing sessions; max |K delta|={max_k:.6g} px; max |dist delta|={max_dist:.6g}')
    print('Original parameters and differences are preserved in conversion.json.')
  print(f"Prepared {len(result['frames'])} groups from {len(exports)} sessions; master={result['master_serial']}; output={args.output}")


if __name__ == '__main__':
  main()
