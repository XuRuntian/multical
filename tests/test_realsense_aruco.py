import json
from pathlib import Path
import pickle
import sys
import tempfile
import unittest

import cv2
import numpy as np
from structs.struct import struct

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from prepare_realsense import prepare, read_json, write_json
from calibrate_realsense import calibrate, export_extrinsics
from multical.board.aruco import ArucoBoard
from multical.board import load_config
from multical.camera_realsense import RealSenseInverseCamera, MODEL
from multical.io.import_calib import import_camera
from multical.io.export_calib import export_camera
from multical.transform import matrix
from multical.workspace import Workspace
from multical import tables


def camera():
  return RealSenseInverseCamera((640, 480),
      np.array([[387., 0, 317.], [0, 386., 246.], [0, 0, 1.]]),
      np.array([-.055, .065, -.00012, .00062, -.021]))


class BoardTests(unittest.TestCase):
  def test_detect_and_pickle(self):
    board = ArucoBoard(.1)
    board = pickle.loads(pickle.dumps(board))
    detections = board.detect(board.draw(pixels_mm=3))
    self.assertTrue(board.has_min_detections(detections))
    np.testing.assert_array_equal(detections.ids, np.arange(4))
    np.testing.assert_allclose(np.linalg.norm(np.roll(board.points, -1, axis=0)-board.points, axis=1), .1)
    self.assertFalse(board.has_min_detections(board.detect(np.full((300, 300), 255, np.uint8))))
    wrong = ArucoBoard(.1, marker_id=1).draw(pixels_mm=3)
    self.assertFalse(board.has_min_detections(board.detect(wrong)))

  def test_pose(self):
    board, cam = ArucoBoard(.1), camera()
    rvec = np.array([.25, -.18, .08])
    tvec = np.array([-.05, -.04, .75])
    points = board.points @ cv2.Rodrigues(rvec)[0].T + tvec
    detections = struct(corners=cam.project(points), ids=board.ids)
    pose, error = board.estimate_pose_points(cam, detections)
    np.testing.assert_allclose(pose[:3], rvec, atol=1e-5)
    np.testing.assert_allclose(pose[3:], tvec, atol=1e-5)
    self.assertLess(error, 1e-3)

  def test_yaml(self):
    path = Path(__file__).resolve().parents[1] / 'example_boards/aruco_100mm.yaml'
    self.assertIsInstance(load_config(str(path))['aruco_100mm'], ArucoBoard)


class CameraTests(unittest.TestCase):
  def test_matches_librealsense_deprojection(self):
    # Reference values queried from the acquisition environment's SDK.
    pixels = np.array([[0, 0], [639, 479], [375, 325], [317, 246]])
    expected = np.array([[-.8286213874816895, -.6440264582633972],
                         [.8391078114509583, .6093720197677612],
                         [.15030533075332642, .20531940460205078], [0, 0]])
    cam = camera()
    actual = (cam.undistort_points(pixels)-cam.principle_point)/cam.focal_length
    np.testing.assert_allclose(actual, expected, atol=2e-7, rtol=0)
    rays = np.concatenate([expected, np.ones((4, 1))], axis=1)
    np.testing.assert_allclose(cam.project(rays), pixels, atol=1e-4, rtol=0)

  def test_roundtrip_serialization(self):
    cam = camera()
    restored = import_camera(export_camera(cam))
    self.assertIsInstance(restored, RealSenseInverseCamera)
    self.assertEqual(restored.copy().model, MODEL)
    np.testing.assert_allclose(restored.dist, cam.dist)
    restored = pickle.loads(pickle.dumps(restored))
    np.testing.assert_allclose(restored.project(np.array([[.1, .2, 1.]])), cam.project(np.array([[.1, .2, 1.]])))


class ConversionTests(unittest.TestCase):
  def make_export(self, root, swapped=False):
    root.mkdir()
    ids = {'111': 'cam01' if swapped else 'cam00', '222': 'cam00' if swapped else 'cam01'}
    intrinsics = dict(width=640, height=480, fx=387., fy=386., ppx=317., ppy=246.,
                      model='distortion.inverse_brown_conrady', coeffs=[-.055, .065, -.00012, .00062, -.021])
    write_json(root/'export_manifest.json', dict(status='complete', camera_ids=ids, exported_groups=3))
    rows = []
    for serial, name in ids.items():
      (root/name).mkdir()
      write_json(root/name/'calibration.json', dict(serial=serial, id=name,
                 original_calibration=dict(intrinsics=dict(color=intrinsics))))
      for i in range(3):
        cv2.imwrite(str(root/name/f'color_{i:06d}.jpg'), np.full((480, 640, 3), 127, np.uint8))
        cv2.imwrite(str(root/name/f'depth_{i:06d}.png'), np.ones((480, 640), np.uint16))
    for i in range(3):
      rows.append(dict(frame_id=i, cameras={name:dict(serial=s, color_path=f'{name}/color_{i:06d}.jpg') for s, name in ids.items()}))
    (root/'frames.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    return root

  def test_merge_identity_and_selection(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      a, b = self.make_export(root/'a'), self.make_export(root/'b', swapped=True)
      out = root/'out'
      result = prepare([a, b], out, master='222')
      self.assertEqual(result['status'], 'complete')
      self.assertEqual([f['source_frame_id'] for f in result['frames']], [1, 1])
      self.assertEqual(sorted(p.name for p in (out/'111').iterdir()), ['pose_000000.jpg', 'pose_000001.jpg'])
      self.assertEqual((out/'111/pose_000001.jpg').resolve(), b/'cam01/color_000001.jpg')
      self.assertEqual(read_json(out/'intrinsics.json')['cameras']['222']['model'], MODEL)
      with self.assertRaises(FileExistsError):
        prepare([a], out, master='222')

  def test_all_groups_and_reject_duplicates(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      a = self.make_export(root/'a')
      self.assertEqual(len(prepare([a], root/'out', '111', 0)['frames']), 3)
      with self.assertRaisesRegex(ValueError, 'Duplicate session'):
        prepare([a, a], root/'bad', '111')
      self.assertFalse((root/'bad').exists())

  def test_parent_directory_discovers_sessions_in_natural_order(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      data = root/'data'
      data.mkdir()
      for name in ['session_10', 'session_02', 'session_001']:
        session = data/name
        session.mkdir()
        self.make_export(session/'hocap')
      (data/'unrelated').mkdir()
      result = prepare([data], root/'out', '111')
      self.assertEqual([Path(f['export']).parent.name for f in result['frames']],
                       ['session_001', 'session_02', 'session_10'])
      self.assertEqual(len(list((root/'out/111').iterdir())), 3)
      with self.assertRaisesRegex(ValueError, 'Duplicate session'):
        prepare([data, data/'session_001'], root/'duplicate', '111')

  def test_parent_rejects_failed_or_raw_only_sessions(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      session = root/'session_001'
      session.mkdir()
      export = self.make_export(session/'hocap')
      manifest = read_json(export/'export_manifest.json')
      manifest['status'] = 'failed'
      write_json(export/'export_manifest.json', manifest)
      with self.assertRaisesRegex(ValueError, 'not complete'):
        prepare([root], root/'out', '111')
      raw = root/'session_002/raw'
      raw.mkdir(parents=True)
      write_json(raw/'manifest.json', {})
      with self.assertRaisesRegex(ValueError, 'only raw data'):
        prepare([root], root/'out', '111')

  def test_empty_or_missing_parent_fails_before_writing(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      with self.assertRaisesRegex(ValueError, 'No session exports'):
        prepare([root], root/'out', '111')
      with self.assertRaisesRegex(ValueError, 'does not exist'):
        prepare([root/'missing'], root/'out', '111')
      self.assertFalse((root/'out').exists())

  def test_reject_changed_intrinsics_and_incomplete_export(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      a, b = self.make_export(root/'a'), self.make_export(root/'b')
      path = b/'cam00/calibration.json'
      data = read_json(path)
      data['original_calibration']['intrinsics']['color']['fx'] += 1
      write_json(path, data)
      with self.assertRaisesRegex(ValueError, 'Intrinsics/profile changed'):
        prepare([a,b], root/'bad', '111')
      data = read_json(a/'export_manifest.json')
      data['status'] = 'failed'
      write_json(a/'export_manifest.json', data)
      with self.assertRaisesRegex(ValueError, 'not complete'):
        prepare([a], root/'bad', '111')

  def test_first_policy_keeps_first_intrinsics_and_records_differences(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      a, b = self.make_export(root/'a'), self.make_export(root/'b')
      path = b/'cam00/calibration.json'
      data = read_json(path)
      data['original_calibration']['intrinsics']['color']['fx'] += .1
      write_json(path, data)
      original = path.read_text()
      result = prepare([a, b], root/'out', '111', intrinsics_policy='first')
      self.assertEqual(read_json(root/'out/intrinsics.json')['cameras']['111']['K'][0][0], 387.)
      self.assertEqual(result['intrinsics_reference_exports']['111'], str(a))
      self.assertEqual(len(result['intrinsics_overrides']), 1)
      self.assertAlmostEqual(result['intrinsics_overrides'][0]['K_delta'][0][0], .1)
      self.assertEqual(path.read_text(), original)
      self.assertEqual(result['source_calibration']['111'][1]['calibration'], data)

  def test_first_policy_still_rejects_profile_or_model_change(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      a, b = self.make_export(root/'a'), self.make_export(root/'b')
      path = b/'cam00/calibration.json'
      data = read_json(path)
      for key, value in [('width', 1280), ('model', 'distortion.brown_conrady')]:
        modified = json.loads(json.dumps(data))
        modified['original_calibration']['intrinsics']['color'][key] = value
        write_json(path, modified)
        with self.assertRaisesRegex(ValueError, 'cannot override'):
          prepare([a, b], root/'out', '111', intrinsics_policy='first')
        self.assertFalse((root/'out').exists())

  def test_disconnected_capture_fails_without_extrinsics(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      export = self.make_export(root/'export')
      prepare([export], root/'prepared', '111')
      with self.assertRaisesRegex(ValueError, 'disconnected'):
        calibrate(root/'prepared', root/'result', num_threads=1)
      self.assertEqual(read_json(root/'result/report.json')['status'], 'failed')
      self.assertFalse((root/'result/extrinsics.json').exists())

  def test_rendered_capture_end_to_end(self):
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      export = self.make_export(root/'export')
      cam = camera()
      board = ArucoBoard(.1)
      relative = matrix.join(cv2.Rodrigues(np.array([.02, .1, -.03]))[0], np.array([.06, .01, .01]))
      marker = cv2.aruco.drawMarker(board.dictionary, 0, 300)
      for i in range(3):
        pose = matrix.join(cv2.Rodrigues(np.array([.15+i*.1, -.25+i*.1, .04]))[0], np.array([-.05+i*.015, -.04, .5+i*.02]))
        for name, transform in [('cam00', pose), ('cam01', relative@pose)]:
          corners = cam.project(matrix.transform_homog(transform, board.points)).astype(np.float32)
          h = cv2.getPerspectiveTransform(np.array([[0,0], [299,0], [299,299], [0,299]], np.float32), corners)
          image = cv2.warpPerspective(marker, h, (640,480), borderValue=255)
          cv2.imwrite(str(export/name/f'color_{i:06d}.jpg'), image)
      prepare([export], root/'prepared', '222', 0)
      result = calibrate(root/'prepared', root/'result', num_threads=1)
      self.assertEqual(result['status'], 'complete')
      extrinsics = read_json(root/'result/extrinsics.json')['T_master_from_camera']
      np.testing.assert_allclose(extrinsics['222'], np.eye(4), atol=1e-8)
      np.testing.assert_allclose(extrinsics['111'], relative, atol=.02)
      restored = Workspace.load(str(root/'result/calibration.pkl'))
      self.assertIsInstance(restored.latest_calibration.cameras[0], RealSenseInverseCamera)


class OptimizationTests(unittest.TestCase):
  def test_synthetic_fixed_intrinsic_extrinsics(self):
    board, cam = ArucoBoard(.1), camera()
    names = ['111', '222']
    relative = matrix.join(cv2.Rodrigues(np.array([.02, .14, -.03]))[0], np.array([.15, .01, .03]))
    observations = [[], []]
    for i in range(6):
      pose = matrix.join(cv2.Rodrigues(np.array([.1+i*.03, -.2+i*.04, .04]))[0], np.array([-.1+i*.02, -.06, .7+i*.03]))
      for j, transform in enumerate([pose, relative@pose]):
        pixels = cam.project(matrix.transform_homog(transform, board.points))
        observations[j].append([struct(corners=pixels, ids=np.arange(4))])
    with tempfile.TemporaryDirectory() as tmp:
      ws = Workspace(tmp, 'synthetic')
      ws.names = struct(camera=names, board=['marker'], image=[str(i) for i in range(6)])
      ws.boards = [board]
      ws.point_table = tables.make_point_table(observations, [board])
      ws.set_calibration(dict(zip(names, [cam, cam.copy()])))
      ws.initialise_poses()
      calib = ws.calibrate(cameras=False, boards=False, board_poses=False, loss='soft_l1', num_adjustments=1)
      result = export_extrinsics(calib, names, '111')
      np.testing.assert_allclose(result['111'], np.eye(4), atol=1e-8)
      np.testing.assert_allclose(result['222'], np.linalg.inv(relative), atol=1e-4)
      for c in calib.cameras:
        np.testing.assert_array_equal(c.intrinsic, cam.intrinsic)
        np.testing.assert_array_equal(c.dist, cam.dist)


if __name__ == '__main__':
  unittest.main()
