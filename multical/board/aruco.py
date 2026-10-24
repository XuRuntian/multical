from cached_property import cached_property
import cv2
import numpy as np
from structs.struct import struct

from .board import Board
from .common import aruco_config, create_dict, empty_detection, estimate_pose_points, quad_polygons
from multical.optimization.parameters import Parameters


class ArucoBoard(Parameters, Board):
  """A single square marker, with corners ordered TL, TR, BR, BL."""

  def __init__(self, marker_length, marker_id=0, aruco_dict='6X6_50',
               aruco_params=None, adjusted_points=None):
    if not np.isfinite(marker_length) or marker_length <= 0:
      raise ValueError('marker_length must be positive (meters)')
    self.marker_length = marker_length
    self.marker_id = marker_id
    self.aruco_dict = aruco_dict
    self.aruco_params = aruco_params or {}
    if not 0 <= marker_id < len(self.dictionary.bytesList):
      raise ValueError('marker_id is outside the dictionary')
    self.adjusted_points = self.points if adjusted_points is None else adjusted_points

  @cached_property
  def dictionary(self):
    return create_dict(self.aruco_dict, 0)

  @property
  def points(self):
    a = self.marker_length
    return np.array([[0, 0, 0], [a, 0, 0], [a, a, 0], [0, a, 0]], dtype=float)

  @property
  def num_points(self):
    return 4

  @property
  def ids(self):
    return np.arange(4)

  @property
  def size_mm(self):
    return [self.marker_length * 1000] * 2

  @property
  def mesh(self):
    return struct(points=self.adjusted_points, polygons=quad_polygons(np.array([[0, 1, 2, 3]])))

  def draw(self, pixels_mm=1, margin=20):
    size = max(16, round(self.marker_length * 1000 * pixels_mm))
    marker = cv2.aruco.drawMarker(self.dictionary, self.marker_id, size)
    return cv2.copyMakeBorder(marker, margin, margin, margin, margin, cv2.BORDER_CONSTANT, value=255)

  def detect(self, image):
    params = {'cornerRefinementMethod': cv2.aruco.CORNER_REFINE_SUBPIX,
              **self.aruco_params}
    corners, ids, _ = cv2.aruco.detectMarkers(image, self.dictionary, parameters=aruco_config(params))
    if ids is None:
      return empty_detection
    matches = np.flatnonzero(ids.ravel() == self.marker_id)
    if len(matches) > 1:
      raise ValueError('Multiple physical copies of the configured ArUco ID are visible')
    if not len(matches):
      return empty_detection
    return struct(corners=corners[matches[0]].reshape(4, 2), ids=self.ids)

  def has_min_detections(self, detections):
    return np.array_equal(detections.ids, self.ids)

  def estimate_pose_points(self, camera, detections):
    return estimate_pose_points(self, camera, detections)

  def export(self):
    return struct(type='aruco', marker_length=self.marker_length, marker_id=self.marker_id,
                  aruco_dict=self.aruco_dict, aruco_params=self.aruco_params)

  def __eq__(self, other):
    return isinstance(other, ArucoBoard) and self.export() == other.export()

  def __repr__(self):
    return 'ArucoBoard ' + str(self.export())

  @property
  def params(self):
    return self.adjusted_points

  def with_params(self, params):
    return ArucoBoard(self.marker_length, self.marker_id, self.aruco_dict,
                      self.aruco_params, adjusted_points=params)

  def __getstate__(self):
    return {key: getattr(self, key) for key in
            ('marker_length', 'marker_id', 'aruco_dict', 'aruco_params', 'adjusted_points')}
