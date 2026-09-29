#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Predictive arm-to-chassis interference guard.

The URDF joint limits describe actuator travel, not self-collision-free
configurations.  This module approximates the distal arm links with capsules
and the chassis with the same oriented box used by the planar Gazebo model.
It predicts the swept clearance of a velocity command before that command is
published.
"""

from __future__ import print_function

import numpy as np

try:
    import PyKDL
    from kdl_parser_py.urdf import treeFromParam
except ImportError:  # Allows geometry/filter unit tests outside ROS.
    PyKDL = None
    treeFromParam = None


ARM_JOINTS = (
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
)


class KDLArmLinkPoseProvider(object):
    """Return selected arm-link transforms in the sway-link frame."""

    def __init__(
            self,
            base_frame="sway_link",
            end_frame="link6",
            robot_description="/robot_description"):
        if PyKDL is None or treeFromParam is None:
            raise RuntimeError("PyKDL and kdl_parser_py are required")

        ok, tree = treeFromParam(robot_description)
        if not ok:
            raise RuntimeError(
                "failed to build KDL tree from {}".format(
                    robot_description
                )
            )

        self.chain = tree.getChain(base_frame, end_frame)
        self.fk_solver = PyKDL.ChainFkSolverPos_recursive(self.chain)
        self.chain_joint_names = []
        self.segment_numbers = {}

        for segment_index in range(self.chain.getNrOfSegments()):
            segment = self.chain.getSegment(segment_index)
            self.segment_numbers[segment.getName()] = segment_index + 1
            joint = segment.getJoint()
            if joint.getTypeName() != "None":
                self.chain_joint_names.append(joint.getName())

        if tuple(self.chain_joint_names) != ARM_JOINTS:
            raise RuntimeError(
                "unexpected arm KDL order: {}".format(
                    self.chain_joint_names
                )
            )

    def link_transforms(self, joint_positions, link_names):
        joint_positions = np.asarray(joint_positions, dtype=np.float64)
        if joint_positions.shape != (len(ARM_JOINTS),):
            raise ValueError("joint_positions must have shape (6,)")

        kdl_positions = PyKDL.JntArray(len(ARM_JOINTS))
        for index, value in enumerate(joint_positions):
            kdl_positions[index] = float(value)

        transforms = {}
        for link_name in link_names:
            if link_name not in self.segment_numbers:
                raise RuntimeError(
                    "link {} is not in the arm KDL chain".format(link_name)
                )
            frame = PyKDL.Frame()
            result = self.fk_solver.JntToCart(
                kdl_positions,
                frame,
                self.segment_numbers[link_name],
            )
            if result < 0:
                raise RuntimeError(
                    "KDL FK failed for {} with code {}".format(
                        link_name,
                        result,
                    )
                )

            transform = np.eye(4, dtype=np.float64)
            for row in range(3):
                transform[row, 3] = frame.p[row]
                for column in range(3):
                    transform[row, column] = frame.M[row, column]
            transforms[link_name] = transform
        return transforms


class ArmChassisInterferenceGuard(object):
    """Scale arm velocity commands before they enter the chassis keep-out."""

    DEFAULT_LINK_CAPSULES = {
        # Endpoints are expressed in each link's CAD frame.  Radii include a
        # small modelling allowance over the STL cross-section.
        "link3": ((0.0, -0.011, -0.350), (0.0, -0.011, 0.020), 0.060),
        "link4": ((0.0, -0.448, 0.000), (0.0, 0.036, 0.000), 0.060),
        "link5": ((0.0, 0.000, -0.326), (0.0, 0.000, 0.015), 0.045),
        "link6": ((0.0, 0.000, -0.010), (0.0, 0.000, 0.000), 0.006),
    }

    def __init__(
            self,
            pose_provider=None,
            hard_clearance=0.010,
            soft_clearance=0.050,
            prediction_horizon=0.25,
            trajectory_samples=5,
            capsule_samples=17,
            chassis_center=(0.43, 0.0, 0.0),
            chassis_half_extents=(0.56, 0.10, 0.145),
            link_capsules=None,
            robot_description="/robot_description"):
        self.pose_provider = (
            pose_provider
            if pose_provider is not None
            else KDLArmLinkPoseProvider(
                robot_description=robot_description
            )
        )
        self.hard_clearance = float(hard_clearance)
        self.soft_clearance = float(soft_clearance)
        self.prediction_horizon = float(prediction_horizon)
        self.trajectory_samples = int(trajectory_samples)
        self.capsule_samples = int(capsule_samples)
        self.chassis_center = self._vector3(
            chassis_center,
            "chassis_center",
        )
        self.chassis_half_extents = self._vector3(
            chassis_half_extents,
            "chassis_half_extents",
        )
        source_capsules = (
            self.DEFAULT_LINK_CAPSULES
            if link_capsules is None
            else link_capsules
        )
        self.link_capsules = self._prepare_capsules(source_capsules)
        self._validate_parameters()

    def minimum_clearance(self, joint_positions):
        """Return signed capsule-to-chassis clearance and closest link."""
        joint_positions = self._vector6(
            joint_positions,
            "joint_positions",
        )
        transforms = self.pose_provider.link_transforms(
            joint_positions,
            list(self.link_capsules.keys()),
        )

        minimum = float("inf")
        closest_link = "none"
        for link_name, capsule in self.link_capsules.items():
            local_points, radius = capsule
            transform = np.asarray(
                transforms[link_name],
                dtype=np.float64,
            )
            if transform.shape != (4, 4):
                raise ValueError("link transform must have shape (4, 4)")
            world_points = (
                np.dot(local_points, transform[0:3, 0:3].T)
                + transform[0:3, 3]
            )
            point_clearance = self._point_box_distance(world_points)
            clearance = float(np.min(point_clearance) - radius)
            if clearance < minimum:
                minimum = clearance
                closest_link = link_name

        if not np.isfinite(minimum):
            raise RuntimeError("arm-chassis clearance is not finite")
        return minimum, closest_link

    def filter_velocity(self, joint_positions, joint_velocities):
        """Return filtered velocity and diagnostics for one control step."""
        joint_positions = self._vector6(
            joint_positions,
            "joint_positions",
        )
        joint_velocities = self._vector6(
            joint_velocities,
            "joint_velocities",
        )
        current_clearance, current_link = self.minimum_clearance(
            joint_positions
        )
        velocity_norm = float(np.linalg.norm(joint_velocities))

        diagnostics = {
            "reason": "safe",
            "scale": 1.0,
            "current_clearance": current_clearance,
            "predicted_clearance": current_clearance,
            "closest_link": current_link,
            "predicted_link": current_link,
        }
        if velocity_norm <= 1.0e-12:
            return joint_velocities.copy(), diagnostics

        end_positions = (
            joint_positions
            + self.prediction_horizon * joint_velocities
        )
        end_clearance, _ = self.minimum_clearance(end_positions)
        predicted_clearance, predicted_link = (
            self._trajectory_minimum_clearance(
                joint_positions,
                joint_velocities,
                1.0,
            )
        )
        diagnostics["predicted_clearance"] = predicted_clearance
        diagnostics["predicted_link"] = predicted_link

        # When already inside the hard envelope, only allow commands whose
        # endpoint increases clearance.  This creates a deterministic escape
        # direction instead of trapping the arm at the boundary.
        if current_clearance <= self.hard_clearance:
            if (
                    end_clearance > current_clearance + 1.0e-5
                    and predicted_clearance
                    >= current_clearance - 1.0e-5):
                diagnostics["reason"] = "chassis_interference_recovery"
                return joint_velocities.copy(), diagnostics
            diagnostics["reason"] = "chassis_interference_hard"
            diagnostics["scale"] = 0.0
            return np.zeros(6, dtype=np.float64), diagnostics

        if predicted_clearance < self.hard_clearance:
            scale = self._maximum_safe_scale(
                joint_positions,
                joint_velocities,
            )
            diagnostics["reason"] = "chassis_interference_hard"
            diagnostics["scale"] = scale
            return joint_velocities * scale, diagnostics

        if (
                predicted_clearance < self.soft_clearance
                and predicted_clearance < current_clearance - 1.0e-5):
            denominator = self.soft_clearance - self.hard_clearance
            scale = (
                predicted_clearance - self.hard_clearance
            ) / denominator
            scale = float(np.clip(scale, 0.0, 1.0))
            diagnostics["reason"] = "chassis_interference_soft"
            diagnostics["scale"] = scale
            return joint_velocities * scale, diagnostics

        return joint_velocities.copy(), diagnostics

    def _trajectory_minimum_clearance(
            self,
            joint_positions,
            joint_velocities,
            velocity_scale):
        minimum = float("inf")
        closest_link = "none"
        for fraction in np.linspace(
                1.0 / self.trajectory_samples,
                1.0,
                self.trajectory_samples):
            candidate = (
                joint_positions
                + fraction
                * self.prediction_horizon
                * velocity_scale
                * joint_velocities
            )
            clearance, link_name = self.minimum_clearance(candidate)
            if clearance < minimum:
                minimum = clearance
                closest_link = link_name
        return minimum, closest_link

    def _maximum_safe_scale(self, joint_positions, joint_velocities):
        lower = 0.0
        upper = 1.0
        for _ in range(18):
            middle = 0.5 * (lower + upper)
            clearance, _ = self._trajectory_minimum_clearance(
                joint_positions,
                joint_velocities,
                middle,
            )
            if clearance >= self.hard_clearance:
                lower = middle
            else:
                upper = middle
        # Stay slightly inside the numerically safe side of the boundary.
        return float(np.clip(lower * 0.98, 0.0, 1.0))

    def _point_box_distance(self, points):
        delta = (
            np.abs(points - self.chassis_center)
            - self.chassis_half_extents
        )
        outside = np.maximum(delta, 0.0)
        return np.linalg.norm(outside, axis=1)

    def _prepare_capsules(self, source_capsules):
        result = {}
        for link_name, values in source_capsules.items():
            start, end, radius = values
            start = self._vector3(start, "capsule start")
            end = self._vector3(end, "capsule end")
            radius = float(radius)
            if radius <= 0.0:
                raise ValueError("capsule radius must be positive")
            fractions = np.linspace(0.0, 1.0, self.capsule_samples)
            points = (
                start[np.newaxis, :]
                + fractions[:, np.newaxis]
                * (end - start)[np.newaxis, :]
            )
            result[str(link_name)] = (points, radius)
        if not result:
            raise ValueError("at least one link capsule is required")
        return result

    def _validate_parameters(self):
        if self.hard_clearance < 0.0:
            raise ValueError("hard_clearance must be non-negative")
        if self.soft_clearance <= self.hard_clearance:
            raise ValueError("soft_clearance must exceed hard_clearance")
        if self.prediction_horizon <= 0.0:
            raise ValueError("prediction_horizon must be positive")
        if self.trajectory_samples <= 0 or self.capsule_samples < 2:
            raise ValueError("invalid guard sampling count")
        if np.any(self.chassis_half_extents <= 0.0):
            raise ValueError("chassis_half_extents must be positive")

    @staticmethod
    def _vector3(value, name):
        vector = np.asarray(value, dtype=np.float64)
        if vector.shape != (3,) or not np.all(np.isfinite(vector)):
            raise ValueError("{} must be a finite 3-vector".format(name))
        return vector.copy()

    @staticmethod
    def _vector6(value, name):
        vector = np.asarray(value, dtype=np.float64)
        if vector.shape != (6,) or not np.all(np.isfinite(vector)):
            raise ValueError("{} must be a finite 6-vector".format(name))
        return vector.copy()
