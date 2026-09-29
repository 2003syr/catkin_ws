#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Convert a 360-degree laser scan into five fixed planar sectors."""

from __future__ import print_function

import math

import numpy as np


SECTOR_NAMES = (
    "front",
    "left_front",
    "left_rear",
    "right_rear",
    "right_front",
)
SECTOR_CENTERS = np.deg2rad(
    np.asarray([0.0, 72.0, 144.0, -144.0, -72.0])
)
DEFAULT_HALF_WIDTH = math.radians(36.0)


def sectorize_scan(
        ranges,
        angle_min,
        angle_increment,
        range_min,
        range_max,
        output_max=10.0,
        half_width=DEFAULT_HALF_WIDTH):
    """Return minimum valid range in five sectors covering the full circle."""
    ranges = np.asarray(ranges, dtype=np.float64)
    if ranges.ndim != 1 or ranges.size == 0:
        raise ValueError("ranges must be a non-empty one-dimensional array")
    angle_increment = float(angle_increment)
    if not np.isfinite(angle_increment) or angle_increment == 0.0:
        raise ValueError("angle_increment must be finite and non-zero")
    range_min = float(range_min)
    range_max = float(range_max)
    output_max = float(output_max)
    half_width = float(half_width)
    if output_max <= 0.0:
        raise ValueError("output_max must be positive")
    if half_width <= 0.0 or half_width > math.pi:
        raise ValueError("half_width must be in (0, pi]")

    effective_max = output_max
    if np.isfinite(range_max) and range_max > 0.0:
        effective_max = min(effective_max, range_max)
    angles = float(angle_min) + np.arange(ranges.size) * angle_increment
    valid_ranges = (
        np.isfinite(ranges)
        & (ranges >= max(0.0, range_min))
    )
    if np.isfinite(range_max) and range_max > 0.0:
        valid_ranges &= ranges <= range_max

    sectors = []
    for center in SECTOR_CENTERS:
        angular_error = np.arctan2(
            np.sin(angles - center),
            np.cos(angles - center),
        )
        mask = (np.abs(angular_error) <= half_width) & valid_ranges
        if np.any(mask):
            distance = float(np.min(ranges[mask]))
            sectors.append(min(distance, effective_max))
        else:
            sectors.append(effective_max)
    return np.asarray(sectors, dtype=np.float32)


def bin_scan(
        ranges,
        angle_min,
        angle_increment,
        range_min,
        range_max,
        output_max=10.0,
        bin_count=36):
    """Return fixed-angle minimum-range bins covering the full circle.

    Bin zero starts at -pi and the last bin ends at +pi.  Invalid and
    uncovered bins use the effective maximum range.  This preserves much
    more obstacle geometry than the five-sector observation without changing
    the legacy 46-D flat ROS observation.
    """
    ranges = np.asarray(ranges, dtype=np.float64)
    if ranges.ndim != 1 or ranges.size == 0:
        raise ValueError("ranges must be a non-empty one-dimensional array")
    angle_increment = float(angle_increment)
    if not np.isfinite(angle_increment) or angle_increment == 0.0:
        raise ValueError("angle_increment must be finite and non-zero")
    output_max = float(output_max)
    bin_count = int(bin_count)
    if output_max <= 0.0:
        raise ValueError("output_max must be positive")
    if bin_count <= 0:
        raise ValueError("bin_count must be positive")

    effective_max = output_max
    if np.isfinite(range_max) and float(range_max) > 0.0:
        effective_max = min(effective_max, float(range_max))
    angles = (
        float(angle_min)
        + np.arange(ranges.size, dtype=np.float64) * angle_increment
    )
    wrapped_angles = np.arctan2(np.sin(angles), np.cos(angles))
    valid = (
        np.isfinite(ranges)
        & (ranges >= max(0.0, float(range_min)))
    )
    if np.isfinite(range_max) and float(range_max) > 0.0:
        valid &= ranges <= float(range_max)

    bins = np.full(bin_count, effective_max, dtype=np.float64)
    if np.any(valid):
        valid_ranges = np.minimum(ranges[valid], effective_max)
        indices = np.floor(
            (wrapped_angles[valid] + math.pi)
            * float(bin_count)
            / (2.0 * math.pi)
        ).astype(np.int64)
        indices = np.clip(indices, 0, bin_count - 1)
        for index, distance in zip(indices, valid_ranges):
            if distance < bins[index]:
                bins[index] = distance
    return bins.astype(np.float32)
