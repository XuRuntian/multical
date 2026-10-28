import json
from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from prepare_segmented_realsense import prepare, read_json, write_json


def intrinsics(fx=50.):
  return dict(width=16, height=12, fx=fx, fy=51., ppx=8., ppy=6.,
              model='distortion.inverse_brown_conrady', coeffs=[0., 0., 0., 0., 0.])


def make_capture(root, fx=50., poses=3):
  root.mkdir()
  ids = {'111': 'cam00', '222': 'cam01'}
  devices = []
  reports = {}
  for serial, camera_id in ids.items():
    devices.append(dict(serial=serial, id=camera_id))
    reports[serial] = dict(calibration=dict(intrinsics=dict(color=intrinsics(fx))))
  write_json(root / 'manifest.json', dict(
    capture_mode='segmented', status='complete', errors=[],
    config=dict(devices=devices), devices=reports))
  export = root / 'calibration_images'
  export.mkdir()
  write_json(export / 'manifest.json', dict(
    schema='sync_lab.calibration_images.v1', status='complete',
    source_capture=str(root.resolve()), camera_ids=ids, exported_poses=poses))
  rows = []
  for serial, camera_id in ids.items():
    (export / camera_id).mkdir()
    for pose in range(poses):
      cv2.imwrite(str(export / camera_id / f'{pose:06d}.png'),
                  np.full((12, 16, 3), 20 + pose, np.uint8))
  for pose in range(poses):
    rows.append(dict(pose_index=pose, segment_id=pose, cameras={
      camera_id: dict(serial=serial, path=f'{camera_id}/{pose:06d}.png')
      for serial, camera_id in ids.items()}))
  (export / 'frames.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
  return root


class SegmentedPreparationTests(unittest.TestCase):
  def test_direct_capture_includes_every_pose_and_preserves_png(self):
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      capture = make_capture(root / 'capture')
      result = prepare([capture], root / 'out', master='222')
      self.assertEqual(result['status'], 'complete')
      self.assertEqual(len(result['frames']), 3)
      self.assertEqual(sorted(path.name for path in (root / 'out/111').iterdir()),
                       ['pose_000000.png', 'pose_000001.png', 'pose_000002.png'])
      self.assertEqual((root / 'out/111/pose_000001.png').resolve(),
                       capture / 'calibration_images/cam00/000001.png')
      self.assertEqual(read_json(root / 'out/intrinsics.json')['cameras']['111']['K'][0][0], 50.)

  def test_parent_discovery_and_first_intrinsics_policy(self):
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      data = root / 'data'
      data.mkdir()
      make_capture(data / 'pose_set_02', fx=50.1, poses=1)
      make_capture(data / 'pose_set_01', fx=50., poses=2)
      with self.assertRaisesRegex(ValueError, 'Intrinsics changed'):
        prepare([data], root / 'strict', master='111')
      result = prepare([data], root / 'first', master='111', intrinsics_policy='first')
      self.assertEqual(len(result['frames']), 3)
      self.assertEqual(len(result['intrinsics_overrides']), 2)
      self.assertEqual(result['frames'][0]['source_pose_index'], 0)
      self.assertEqual(read_json(root / 'first/intrinsics.json')['cameras']['111']['K'][0][0], 50.)

  def test_rejects_incomplete_or_changed_source_identity(self):
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      capture = make_capture(root / 'capture')
      manifest_path = capture / 'calibration_images/manifest.json'
      manifest = read_json(manifest_path)
      manifest['status'] = 'failed'
      write_json(manifest_path, manifest)
      with self.assertRaisesRegex(ValueError, 'not complete'):
        prepare([capture], root / 'out', master='111')
      self.assertFalse((root / 'out').exists())


if __name__ == '__main__':
  unittest.main()
