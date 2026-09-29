#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Small rolling occupancy grid built directly from a planar LaserScan."""

from __future__ import print_function

import math

import numpy as np


class LocalOccupancyGrid(object):
    """Accumulate a static local map and construct SE(2) collision layers.

    The map stores occupied laser endpoints in the world frame and clears
    cells along every observed ray.  Unknown space remains traversable, while
    the planner replans as new obstacle surfaces become visible.
    """

    def __init__(
            self,
            resolution=0.05,
            width=5.0,
            height=5.0,
            max_ray_length=3.0,
            occupied_increment=4,
            free_decrement=1,
            occupied_threshold=2,
            minimum_log_odds=-8,
            maximum_log_odds=12):
        self.resolution = float(resolution)
        self.width = float(width)
        self.height = float(height)
        self.max_ray_length = float(max_ray_length)
        self.occupied_increment = int(occupied_increment)
        self.free_decrement = int(free_decrement)
        self.occupied_threshold = int(occupied_threshold)
        self.minimum_log_odds = int(minimum_log_odds)
        self.maximum_log_odds = int(maximum_log_odds)
        self._validate()

        self.width_cells = int(math.ceil(self.width / self.resolution))
        self.height_cells = int(math.ceil(self.height / self.resolution))
        self.center = np.zeros(2, dtype=np.float64)
        self.origin = np.zeros(2, dtype=np.float64)
        self.log_odds = np.zeros(
            (self.height_cells, self.width_cells),
            dtype=np.int16,
        )
        self.observed = np.zeros_like(self.log_odds, dtype=np.bool_)
        self.scan_updates = 0
        self.reset((0.0, 0.0))

    def reset(self, center):
        center = self._vector(center, 2, "center")
        self.center = center.copy()
        self.origin = center - np.asarray([
            0.5 * self.width,
            0.5 * self.height,
        ], dtype=np.float64)
        self.log_odds.fill(0)
        self.observed.fill(False)
        self.scan_updates = 0

    def update_scan(
            self,
            pose,
            ranges,
            angle_min,
            angle_increment,
            range_min,
            range_max):
        """Fuse one scan using a body-forward pose [x, y, heading]."""
        pose = self._vector(pose, 3, "pose")
        ranges = np.asarray(ranges, dtype=np.float64)
        if ranges.ndim != 1 or ranges.size == 0:
            raise ValueError("ranges must be a non-empty vector")
        angle_increment = float(angle_increment)
        if not np.isfinite(angle_increment) or angle_increment == 0.0:
            raise ValueError("angle_increment must be finite and non-zero")

        sensor_cell = self.world_to_grid(pose[0:2])
        if sensor_cell is None:
            return
        finite_max = float(range_max)
        if not np.isfinite(finite_max) or finite_max <= 0.0:
            finite_max = self.max_ray_length
        usable_max = min(self.max_ray_length, finite_max)
        minimum = max(0.0, float(range_min))
        angles = (
            float(angle_min)
            + np.arange(ranges.size, dtype=np.float64) * angle_increment
            + pose[2]
        )

        for distance, angle in zip(ranges, angles):
            valid_hit = bool(
                np.isfinite(distance)
                and distance >= minimum
                and distance < finite_max - 1.0e-3
                and distance <= self.max_ray_length
            )
            ray_distance = (
                min(float(distance), usable_max)
                if np.isfinite(distance) and distance >= minimum
                else usable_max
            )
            endpoint = pose[0:2] + ray_distance * np.asarray([
                math.cos(angle),
                math.sin(angle),
            ])
            endpoint_cell = self.world_to_grid(endpoint, clip=True)
            cells = self._bresenham(sensor_cell, endpoint_cell)
            if not cells:
                continue
            free_cells = cells[:-1] if valid_hit else cells
            for ix, iy in free_cells:
                self.observed[iy, ix] = True
                self.log_odds[iy, ix] = max(
                    self.minimum_log_odds,
                    int(self.log_odds[iy, ix]) - self.free_decrement,
                )
            if valid_hit:
                ix, iy = cells[-1]
                self.observed[iy, ix] = True
                self.log_odds[iy, ix] = min(
                    self.maximum_log_odds,
                    int(self.log_odds[iy, ix])
                    + self.occupied_increment,
                )
        self.scan_updates += 1

    def occupied_mask(self):
        return self.log_odds >= self.occupied_threshold

    def configuration_obstacles(
            self,
            heading_bins,
            footprint_half_length,
            footprint_half_width,
            safety_margin):
        """Return one full-footprint collision grid per heading bin."""
        heading_bins = int(heading_bins)
        half_length = (
            float(footprint_half_length) + float(safety_margin)
        )
        half_width = (
            float(footprint_half_width) + float(safety_margin)
        )
        if heading_bins <= 0:
            raise ValueError("heading_bins must be positive")
        if half_length <= 0.0 or half_width <= 0.0:
            raise ValueError("inflated footprint must be positive")

        occupied = self.occupied_mask()
        layers = np.zeros(
            (heading_bins, self.height_cells, self.width_cells),
            dtype=np.bool_,
        )
        radius_cells = int(math.ceil(
            math.hypot(half_length, half_width) / self.resolution
        ))
        for heading_index in range(heading_bins):
            heading = (
                -math.pi
                + 2.0 * math.pi
                * float(heading_index)
                / float(heading_bins)
            )
            cosine = math.cos(heading)
            sine = math.sin(heading)
            layer = layers[heading_index]
            for dy in range(-radius_cells, radius_cells + 1):
                for dx in range(-radius_cells, radius_cells + 1):
                    world_dx = float(dx) * self.resolution
                    world_dy = float(dy) * self.resolution
                    body_x = cosine * world_dx + sine * world_dy
                    body_y = -sine * world_dx + cosine * world_dy
                    if (
                            abs(body_x) <= half_length
                            and abs(body_y) <= half_width):
                        self._shift_or(layer, occupied, -dx, -dy)

            # Keep the complete footprint inside the local planning window.
            layer[:radius_cells, :] = True
            layer[-radius_cells:, :] = True
            layer[:, :radius_cells] = True
            layer[:, -radius_cells:] = True
        return layers

    def world_to_grid(self, point, clip=False):
        point = self._vector(point, 2, "point")
        cell = np.floor(
            (point - self.origin) / self.resolution
        ).astype(np.int64)
        ix = int(cell[0])
        iy = int(cell[1])
        if clip:
            ix = int(np.clip(ix, 0, self.width_cells - 1))
            iy = int(np.clip(iy, 0, self.height_cells - 1))
            return ix, iy
        if (
                ix < 0 or ix >= self.width_cells
                or iy < 0 or iy >= self.height_cells):
            return None
        return ix, iy

    def grid_to_world(self, cell):
        ix, iy = int(cell[0]), int(cell[1])
        return self.origin + self.resolution * np.asarray([
            float(ix) + 0.5,
            float(iy) + 0.5,
        ])

    def occupancy_message_values(self):
        values = np.full(self.log_odds.shape, -1, dtype=np.int8)
        values[self.observed] = 0
        values[self.occupied_mask()] = 100
        return values

    @staticmethod
    def _shift_or(destination, source, shift_x, shift_y):
        height, width = source.shape
        source_x0 = max(0, -shift_x)
        source_x1 = min(width, width - shift_x)
        source_y0 = max(0, -shift_y)
        source_y1 = min(height, height - shift_y)
        if source_x0 >= source_x1 or source_y0 >= source_y1:
            return
        destination_x0 = source_x0 + shift_x
        destination_x1 = source_x1 + shift_x
        destination_y0 = source_y0 + shift_y
        destination_y1 = source_y1 + shift_y
        destination[
            destination_y0:destination_y1,
            destination_x0:destination_x1,
        ] |= source[source_y0:source_y1, source_x0:source_x1]

    @staticmethod
    def _bresenham(start, end):
        x0, y0 = int(start[0]), int(start[1])
        x1, y1 = int(end[0]), int(end[1])
        cells = []
        dx = abs(x1 - x0)
        sx = 1 if x0 < x1 else -1
        dy = -abs(y1 - y0)
        sy = 1 if y0 < y1 else -1
        error = dx + dy
        while True:
            cells.append((x0, y0))
            if x0 == x1 and y0 == y1:
                break
            twice_error = 2 * error
            if twice_error >= dy:
                error += dy
                x0 += sx
            if twice_error <= dx:
                error += dx
                y0 += sy
        return cells

    def _validate(self):
        if self.resolution <= 0.0:
            raise ValueError("resolution must be positive")
        if self.width <= self.resolution or self.height <= self.resolution:
            raise ValueError("map dimensions are too small")
        if self.max_ray_length <= 0.0:
            raise ValueError("max_ray_length must be positive")
        if self.occupied_increment <= 0 or self.free_decrement <= 0:
            raise ValueError("occupancy increments must be positive")

    @staticmethod
    def _vector(value, size, name):
        vector = np.asarray(value, dtype=np.float64)
        if vector.shape != (size,) or not np.all(np.isfinite(vector)):
            raise ValueError(
                "{} must be a finite shape-{} vector".format(name, size)
            )
        return vector
