"""RealSense inverse Brown-Conrady, matching librealsense rs.cpp deprojection.

The inverse model's tangential terms differ from OpenCV's standard model.
Keep source pixels and coefficients unchanged throughout optimization.
"""
from cached_property import cached_property
import numpy as np

from .camera import Camera


MODEL = 'realsense_inverse_brown_conrady'


class RealSenseInverseCamera(Camera):
  def __init__(self, image_size, intrinsic, dist, model=MODEL, **kwargs):
    if model != MODEL:
      raise ValueError('Unexpected RealSense distortion model')
    super().__init__(image_size, intrinsic, dist, model='standard', **kwargs)
    if np.size(self.dist) != 5 or not np.all(np.isfinite(self.dist)):
      raise ValueError('RealSense inverse model requires five finite coefficients')
    self.dist = np.asarray(self.dist, dtype=float).reshape(5)
    self.model = MODEL

  def _distort(self, xy):
    x, y = xy[..., 0], xy[..., 1]
    k1, k2, p1, p2, k3 = self.dist
    r2 = x*x + y*y
    radial = 1 + r2 * (k1 + r2 * (k2 + r2 * k3))
    xq, yq = x * radial, y * radial
    return np.stack((xq + 2*p1*xq*yq + p2*(r2 + 2*xq*xq),
                     yq + 2*p2*xq*yq + p1*(r2 + 2*yq*yq)), axis=-1)

  def project(self, points):
    points = np.asarray(points)
    xy = points[..., :2] / points[..., 2:3]
    return self._distort(xy) * self.focal_length + self.principle_point

  def undistort_points(self, points):
    xy0 = (np.asarray(points) - self.principle_point) / self.focal_length
    xy = xy0.copy()
    k1, k2, p1, p2, k3 = self.dist
    # Same ten fixed-point steps as rs2_deproject_pixel_to_point.
    for _ in range(10):
      x, y = xy[..., 0], xy[..., 1]
      r2 = x*x + y*y
      radial = 1 + r2 * (k1 + r2 * (k2 + r2*k3))
      xq, yq = x*radial, y*radial
      delta = np.stack((2*p1*xq*yq + p2*(r2 + 2*xq*xq),
                        2*p2*xq*yq + p1*(r2 + 2*yq*yq)), axis=-1)
      xy = (xy0 - delta) / radial[..., None]
    return xy * self.focal_length + self.principle_point

  @cached_property
  def undistort_map(self):
    w, h = self.image_size
    y, x = np.mgrid[:h, :w]
    ideal = (np.stack((x, y), axis=-1) - self.principle_point) / self.focal_length
    return (self._distort(ideal) * self.focal_length + self.principle_point).astype(np.float32)

  def copy(self, **kwargs):
    values = self.__getstate__()
    values.update(kwargs)
    return RealSenseInverseCamera(**values)
